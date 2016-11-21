#
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include(FindCUDA)

set(INC_DIRS /usr/local/cuda/include)
set(LIB_DIRS ${BING++_BINARY_DIR}/LibLinear)

#OPENCV_DIR should be set as an environment variable in the shell if it is not
#present in the standard directories

include_directories(${INC_DIRS})
link_directories(${LIB_DIRS})

add_subdirectory(ORUtils)
add_subdirectory(FelzenSegment)
add_subdirectory(BSDSBench)

set(GSLIC_LIB
gSLIC/engines/gSLIC_core_engine.h
gSLIC/engines/gSLIC_seg_engine.h
gSLIC/engines/gSLIC_seg_engine_GPU.h
gSLIC/engines/gSLIC_seg_engine_shared.h
gSLIC/engines/gSLIC_core_engine.cpp
gSLIC/engines/gSLIC_seg_engine.cpp
gSLIC/engines/gSLIC_seg_engine_GPU.cu
gSLIC/objects/gSLIC_settings.h
gSLIC/objects/gSLIC_spixel_info.h
gSLIC/gSLIC_defines.h
gSLIC/gSLIC.h
)

SOURCE_GROUP(engines FILES ${GSLIC_LIB})

if (${CMAKE_COMPILER_IS_GNUCXX})
  set(OPENMP_FLAG "-fopenmp -std=c++11 -ftree-vectorize")
endif (${CMAKE_COMPILER_IS_GNUCXX})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG} -O3")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50)

set(SRC_FILES CmFile.cpp HFSSegment.cpp Magnitude.cu Texture.cpp)
cuda_add_library(gSLIC_lib ${GSLIC_LIB})
add_library(Segment ${SRC_FILES})
cuda_add_executable(${PROJECT_NAME} ${SRC_FILES} kernel.cu)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARY} liblinear gSLIC_lib FelzenSegment BSDSBench)
