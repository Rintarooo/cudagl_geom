cmake_minimum_required(VERSION 2.8)
project(GPU_MC)

# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# add_subdirectory(OpenCLUtilities)
# include_directories(${OPENCL_INCLUDE_DIRS})

find_package(CUDA 10.1 REQUIRED)

find_package(GLEW REQUIRED)

find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)

include_directories(
	${OPENGL_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIR}
	${GLEW_INCLUDE_DIR}
	include
)
cuda_add_executable(main Metaball.cpp BasicGeom.cpp Display.cpp MarhingCubes.cpp Metaball_kernel.cu)

target_link_libraries(main 
	${GLUT_LIBRARIES}
	${OPENGL_LIBRARIES}
	${CUDA_LIBRARIES}
	${GLEW_LIBRARIES}
	# ${GLEW_LIBRARY}
)
  
