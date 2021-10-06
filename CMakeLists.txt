cmake_minimum_required(VERSION 2.8)
project(MarchingCube)

find_package(CUDA 10.1 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
	${CUDA_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
	${GLUT_INCLUDE_DIR}
	${GLEW_INCLUDE_DIR}
	include
)

cuda_add_executable(main Metaball.cpp BasicGeom.cpp Display.cpp MarhingCubes.cpp Metaball_kernel.cu)

target_link_libraries(main 
	${CUDA_LIBRARIES}
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${GLEW_LIBRARIES}
)
  
