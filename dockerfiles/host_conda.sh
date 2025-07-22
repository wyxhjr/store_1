conda deactivate
conda remove -y --name cheapg --all
conda create -y -n cheapg python=3.9
conda activate cheapg
conda install -y -c nvidia/label/cuda-12.1.1 cuda cuda-toolkit cuda-libraries cuda-cudart
conda install -y -c conda-forge folly tbb pybind11
conda install -y mkl mkl-include
# conda install -y conda-forge/label/gcc7::gcc_linux-64
