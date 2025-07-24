## Prepare the environment to run SGLang on CPU

**GCC >= 11 is needed**

```sh
# Create conda ENV
conda create -n sglang python=3.10
conda activate sglang

# Install SGLang
git clone https://github.com/jianan-gu/sglang.git

cd sglang

git checkout mlperf_support_v2

pip install -e "python[all_cpu]"

conda install -y libsqlite=3.48.0

pip uninstall torch torchvision
pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.9.0.dev20250701%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl

# Build sgl-kernel
conda install -y libnuma numactl

cd sgl-kernel
pip install uv
pip install scikit-build-core
cp pyproject_cpu.toml pyproject.toml
uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
pip install dist/*.whl --force-reinstall
cd ..

conda install -y gperftools -c conda-forge
pip install intel-openmp==2024.2.0
```
## Example CMD
**The following command lines are for demonstration purposes. Modify the arguments and thread binding according to your requirements and CPU type.**

**Please avoid cross NUMA node memory access when setting SGLANG_CPU_OMP_THREADS_BIND.**

`SGLANG_CPU_OMP_THREADS_BIND` specifies the CPU cores dedicated to the OpenMP threads. `--tp` sets the TP size. Below are the example of running without TP and with TP = 6. By changing `--tp` and `SGLANG_CPU_OMP_THREADS_BIND` accordingly, you could set TP size to other values and specifiy the core binding for each rank.

**Preload iomp and tcmalloc for better performance.**
```sh
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so
```
### Bench one batch
#### llama3.1 8B INT4 (DA16W4)
```sh
SGLANG_USE_CPU_ENGINE=1 SGLANG_CPU_OMP_THREADS_BIND="0-39|40-79" python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 1024  --model /Path/to/llama8b_int4/   --trust-remote-code --device cpu  --max-total-tokens 65536 --mem-fraction-static 0.8  --tp 2
```

#### llama3.1 8B INT4 (DA8W4)
```sh
SGLANG_USE_CPU_ENGINE=1 SGLANG_USE_CPU_INT4_W4A8=1 SGLANG_CPU_OMP_THREADS_BIND="0-39|40-79" python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 1024  --model /Path/to/llama8b_int4/ --trust-remote-code --device cpu  --max-total-tokens 65536 --mem-fraction-static 0.8  --tp 2
```

