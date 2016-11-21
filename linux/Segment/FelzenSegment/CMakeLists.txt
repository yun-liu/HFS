#############################
# Specify the project files #
#############################

##
SET(FELZENSEGMENT_HEADERS
convolve.h
disjoint-set.h
filter.h
image.h
segment-image.h
)

#################################################################
# Collect the project files into common, CPU-only and CUDA-only #
#################################################################

set(FELZENSEGMENT_OBJECTS
segment-image.cpp
${FELZENSEGMENT_HEADERS}
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${FELZENSEGMENT_HEADERS})

##############################################################
# Specify the include directories, target and link libraries #
##############################################################

add_library(FelzenSegment ${FELZENSEGMENT_OBJECTS})
