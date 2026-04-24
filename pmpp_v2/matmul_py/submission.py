#!POPCORN leaderboard matmul_v2
#!POPCORN gpu B200

# Direct cuBLASlt matmul with algo heuristic + shape cache.
# Wins over at::matmul_out by:
#  - bypassing ATen dispatch layers
#  - explicitly picking the best algo via cublasLtMatmulAlgoGetHeuristic
#  - allowing a large (32MB) workspace so cuBLAS can select faster algos
#  - caching the algo per shape so we only pay the search cost once per shape
#
# Falls back to cuBLAS via at::matmul_out if anything fails.
from task import input_t, output_t
import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False

from torch.utils.cpp_extension import load_inline


_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cublasLt.h>
#include <cstdint>
#include <unordered_map>

// Single persistent cuBLASlt handle.
static cublasLtHandle_t g_handle = nullptr;

struct MatmulPlan {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatmulAlgo_t algo = {};
    size_t ws_bytes = 0;
    bool has_algo = false;
};

// Cache plans by (M, N, K, dtype_code) packed as uint64.
static std::unordered_map<uint64_t, MatmulPlan> g_cache;

static inline uint64_t pack_key(int M, int N, int K, int dtype_code) {
    // Layout: [20b M | 20b N | 20b K | 4b dtype]
    return ((uint64_t)(M & 0xFFFFF) << 44)
         | ((uint64_t)(N & 0xFFFFF) << 24)
         | ((uint64_t)(K & 0xFFFFF) << 4)
         | (uint64_t)(dtype_code & 0xF);
}

static void build_plan(MatmulPlan& plan,
                       int M, int N, int K,
                       cudaDataType_t abc_dtype,
                       cublasComputeType_t compute_type,
                       size_t max_ws_bytes) {
    // PyTorch row-major A(M,K) @ B(K,N) = C(M,N).
    // In cuBLAS column-major land: C^T(N,M) = B^T(N,K) @ A^T(K,M)
    // We swap A and B in the call and set dimensions m=N, n=M, k=K.
    // All layouts are declared as if the data were column-major storage —
    // which is exactly PyTorch's row-major interpreted as col-major transposed.

    cublasLtMatmulDescCreate(&plan.op_desc, compute_type, CUDA_R_32F);
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &op_n, sizeof(op_n));

    // cuBLAS "A" layout = PyTorch B viewed as col-major: (N, K), lda = N
    cublasLtMatrixLayoutCreate(&plan.a_layout, abc_dtype, N, K, N);
    // cuBLAS "B" layout = PyTorch A viewed as col-major: (K, M), ldb = K
    cublasLtMatrixLayoutCreate(&plan.b_layout, abc_dtype, K, M, K);
    // cuBLAS "C" layout = PyTorch C viewed as col-major: (N, M), ldc = N
    cublasLtMatrixLayoutCreate(&plan.c_layout, abc_dtype, N, M, N);

    // Heuristic algo selection
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_ws_bytes, sizeof(max_ws_bytes));

    cublasLtMatmulHeuristicResult_t heuristic[8];
    int returned = 0;
    auto status = cublasLtMatmulAlgoGetHeuristic(
        g_handle, plan.op_desc,
        plan.a_layout, plan.b_layout, plan.c_layout, plan.c_layout,
        pref, 8, heuristic, &returned);

    if (status == CUBLAS_STATUS_SUCCESS && returned > 0) {
        plan.algo = heuristic[0].algo;
        plan.ws_bytes = heuristic[0].workspaceSize;
        plan.has_algo = true;
    }
    cublasLtMatmulPreferenceDestroy(pref);
}

// Main entry: workspace tensor allocated in Python and passed in.
void matmul_lt(torch::Tensor A, torch::Tensor B, torch::Tensor out,
               torch::Tensor workspace) {
    if (g_handle == nullptr) cublasLtCreate(&g_handle);

    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(1);

    cudaDataType_t dtype;
    int dtype_code;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    switch (A.scalar_type()) {
        case torch::kHalf:
            dtype = CUDA_R_16F;  dtype_code = 0; break;
        case torch::kBFloat16:
            dtype = CUDA_R_16BF; dtype_code = 1; break;
        case torch::kFloat32:
            dtype = CUDA_R_32F;  dtype_code = 2; break;
        default:
            // Unsupported dtype — fall back via ATen
            at::matmul_out(out, A, B);
            return;
    }

    size_t ws_size = (size_t)workspace.numel();
    uint64_t key = pack_key(M, N, K, dtype_code);
    auto it = g_cache.find(key);
    if (it == g_cache.end()) {
        MatmulPlan plan;
        build_plan(plan, M, N, K, dtype, compute_type, ws_size);
        if (!plan.has_algo) {
            // Heuristic didn't return anything; fall back
            at::matmul_out(out, A, B);
            return;
        }
        g_cache[key] = plan;
        it = g_cache.find(key);
    }
    const MatmulPlan& plan = it->second;

    // alpha/beta scalars: cuBLASlt expects the compute type.
    // For CUBLAS_COMPUTE_32F, alpha/beta are float.
    float alpha = 1.0f, beta = 0.0f;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasLtMatmul(g_handle, plan.op_desc,
                    &alpha,
                    B.data_ptr(), plan.a_layout,   // cuBLAS "A" = PyTorch B
                    A.data_ptr(), plan.b_layout,   // cuBLAS "B" = PyTorch A
                    &beta,
                    out.data_ptr(), plan.c_layout,
                    out.data_ptr(), plan.c_layout,
                    &plan.algo,
                    workspace.data_ptr(),
                    plan.ws_bytes,
                    stream);
}
"""

_CPP_SRC = """
void matmul_lt(torch::Tensor A, torch::Tensor B, torch::Tensor out, torch::Tensor workspace);
"""

_mod = load_inline(
    name="matmul_cublaslt_tuned",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["matmul_lt"],
    extra_cuda_cflags=["-O3", "-arch=sm_100"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublasLt", "-lcublas"],
    verbose=False,
)


# 32 MB workspace — large enough to let cuBLAS pick its fastest algos.
_WORKSPACE: dict = {}


def _get_workspace(device):
    ws = _WORKSPACE.get(device)
    if ws is None:
        ws = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        _WORKSPACE[device] = ws
    return ws


def custom_kernel(data: input_t) -> output_t:
    A, B, out = data
    ws = _get_workspace(A.device)
    _mod.matmul_lt(A, B, out, ws)
    return out
