"""Remote GPU runner via Modal. No local GPU needed.

First time:
    pip install modal && modal setup

Run a script on a rented H100 (billed per second, ~$0.001/s idle-to-idle):
    modal run modal/run.py::run --script triton/vector_add.py
    modal run modal/run.py::run --script triton/matmul.py --gpu H100
"""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton==3.1.0", "numpy")
)

app = modal.App("gpumode-scratch", image=image)


@app.function(gpu="A10G", timeout=600)
def _run_remote(script_text: str, script_name: str) -> str:
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, os.path.basename(script_name))
        with open(path, "w") as f:
            f.write(script_text)
        r = subprocess.run(["python", path], capture_output=True, text=True)
        return f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"


@app.local_entrypoint()
def run(script: str, gpu: str = "A10G"):
    p = REPO / script
    assert p.exists(), f"no such file: {p}"
    # Rebuild the function with the requested GPU type at call time.
    f = _run_remote.with_options(gpu=gpu)
    print(f.remote(p.read_text(), p.name))
