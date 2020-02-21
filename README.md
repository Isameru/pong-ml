# Pong: The Machine Learning Game

## Building on Linux

### Requirements

#### C++17 Compiler of Your choice
```bash
sudo apt install g++
```


#### CMake 3.16.4
You may use [Kitware APT Repository](https://apt.kitware.com/).

#### Simple DirectMedia Layer (v2) with *Image* and *True Type Font* Extensions
```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libglew-dev
```

#### OpenGL Extension Wrangler Library
```bash
sudo apt install libglew-dev
```

### Compilation
```bash
git clone https://github.com/Isameru/pong-ml.git
cd pong-ml
mkdir _out
cd _out
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j`nproc`
pong-ml/pong-ml   # To start the game
```

### CUDA Support
If you have the machine capable with strong graphics card and would like to take advantage of *GPGPU*, install [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-downloads).
*CMake* should detect the precesence of this toolkit and build the game using *CUDA*-enabled version of *LibTorch*.