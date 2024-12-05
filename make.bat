set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set NVCUDASAMPLES_ROOT= C:\ProgramData\NVIDIA Corporation\CUDA Samples\v12.6
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp;%PATH%
cmake -S . -B build -DGeant4_DIR=G:\MonteCarlo\geant4-11.2.2-win64\lib\cmake\Geant4