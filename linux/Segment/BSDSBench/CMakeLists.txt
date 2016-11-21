#############################
# Specify the project files #
#############################

##
SET(BSDSBench_HEADERS
Bench.h
source/Array.hh
source/csa.hh
source/csa_defs.h
source/csa_types.h
source/Exception.hh
source/kofn.hh
source/match.hh
source/Matrix.hh
source/Point.hh
source/Random.hh
source/Sort.hh
source/String.hh
)

#################################################################
# Collect the project files into common, CPU-only and CUDA-only #
#################################################################

set(BSDSBench_OBJECTS
AllBench.cpp
BoundaryBench.cpp
RegionBench.cpp
Utils.cpp
source/correspondPixels.cc
source/csa.cc
source/Exception.cc
source/kofn.cc
source/match.cc
source/Matrix.cc
source/Random.cc
source/String.cc
${BSDSBench_HEADERS}
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${BSDSBench_HEADERS})

##############################################################
# Specify the include directories, target and link libraries #
##############################################################

if (${CMAKE_COMPILER_IS_GNUCXX})
  set(OPENMP_FLAG "-fopenmp -std=c++11 -ftree-vectorize")
endif (${CMAKE_COMPILER_IS_GNUCXX})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG} -O3")
add_library(BSDSBench ${BSDSBench_OBJECTS})
