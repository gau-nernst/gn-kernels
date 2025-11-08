import modal

app = modal.App("example-get-started")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.9.0")
    .add_local_python_source("benchmark_sol")
)


@app.function(image=image, gpu="B200")
def benchmark_sol_b200():
    from benchmark_sol import benchmark_sol

    benchmark_sol()


@app.local_entrypoint()
def main():
    benchmark_sol_b200.remote()
