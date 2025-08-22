## How to Build SGlang to run with both CUDA and CPU kernels (conda env)
```
# build default sglang
git clone  https://github.com/jianan-gu/sglang -b heteroflow
cd sglang
# install default CUDA ENVs, this will auto install CUDA version sgl-kernel package
pip install -e "python[all]"

# build and install CPU kernels (sgl-kernel-cpu package)
cd sgl-kernel
pip install uv
pip install scikit-build-core
pip install cmake
cp pyproject_cpu.toml pyproject.toml
uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation
pip install dist/dist/sgl_kernel_cpu-0.3.6.post1-cp310-cp310-linux_x86_64.whl
cd ..

# Test SGlang with both CUDA+CPU kernels UT
python test/srt/cpu/test_heteroflow_kernel_call.py
```
