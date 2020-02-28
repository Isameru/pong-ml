# Pong: The Machine Learning Game

The game of pong made to train a bot.

The current attempt fails to learn using (Double) Deep Q-Learning Network.

**If you know why it does not work - I would be very grateful for letting me know!**

## Building on Linux

### Requirements

#### C++17 Compiler of Your choice
```bash
sudo apt install g++
```

#### CMake 3.16.4
You may use [Kitware APT Repository](https://apt.kitware.com/).

### LibTorch 1.4
Unzip the [PyTorch package](https://pytorch.org/get-started/locally/).
Select *Stable (1.4)* version, *LibTorch* package for *C++/Java*

In `external` directory, make symbolic link to LibTorch:
```bash
cd pong-ml/external
ln -s <path to LibTorch> libtorch-linux
ln -s <path to LibTorch with CUDA support> libtorch-cuda-linux
```

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
If you have the machine capable with strong graphics card and would like to take advantage of *GPGPU*, install [NVIDIA CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-downloads) and [NVIDIA cuDNN v7.6.5](https://developer.nvidia.com/cudnn).
*CMake* should detect the precesence of this toolkit and build the game using *CUDA*-enabled version of *LibTorch*. If you experience issues, hack the `CMakeLists.txt` to satisfy *cmake*, compilation, linking, and run-time errors.
