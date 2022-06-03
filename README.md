source codes are quoted from https://gihyo.jp/book/2014/978-4-7741-6304-8/support

I have modified the original codes above as follows

* run Docker container
* build with CMake

## CUDA + OpenGL

build docker image, OpenGL + CUDA

modify dockerfiles/Dockerfile depending on your environment (such as your desired OpenCV version, Compute Capability(CC) of your GPU).

(building might take some time...)
```bash
docker build -t $(id -un)/cudagl:10.1-ubuntu18.04-opencv3.4.11-CC5.2-pcl1.11.0 ./dockerfiles/cudagl/
```
<br>

run container
```bash
docker run --rm -it --runtime=nvidia --cap-add=SYS_PTRACE --security-opt="seccomp=unconfined" -v $HOME/coding/:/opt -e CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics $(id -un)/cudagl:10.1-ubuntu18.04-opencv3.4.11-CC5.2-pcl1.11.0 
cudagl:10.1-devel-ubuntu18.04-opencv3.4.11-CC5.2-pcl1.11.0
```
<br>


### build and run
```bash
./run.sh
./build/main
```
<br>


If you cannot display images(like bad X server connection. ), you may need to run xhost si:localuser:$USER or worst case xhost local:root before running docker container if get errors like Error: cannot open display

Ref: https://github.com/turlucode/ros-docker-gui

### Reference

* 技術評論社 GPU 並列図形処理入門 - CUDA・OpenGLの導入と活用

