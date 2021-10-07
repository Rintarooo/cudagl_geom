source codes are from https://gihyo.jp/book/2014/978-4-7741-6304-8/support

I have modified the original codes above as follows

* run Docker container
* build with CMake

## CUDA + OpenGL

build docker image, OpenGL + CUDA
```bash
docker build -t $(id -un)/cudagl:10.1-ubuntu18.04-opencv3.4.11-CC5.0-pcl1.11.0 ./dockerfiles/cudagl/
```
<br>

run container
```bash
docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics $(id -un)/cudagl:10.1-ubuntu18.04-opencv3.4.11-CC5.0-pcl1.11.0
```
<br>


### build and run
```bash
./run.sh
./build/main
```
<br>

### Reference

* 技術評論社 GPU 並列図形処理入門 - CUDA・OpenGLの導入と活用

