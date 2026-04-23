NVCC ?= nvcc
# Override for your GPU: 80=A100, 86=RTX30xx, 89=RTX40xx, 90=H100
ARCH ?= 86
NVCCFLAGS = -O3 -arch=sm_$(ARCH) -std=c++17

BINS = vector_add matmul_naive

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean
