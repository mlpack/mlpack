cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   ansi_colors.h
   arma_extend.h
   arma_extend/arma_ostream_prefixed_bones.hpp
   arma_extend/arma_ostream_prefixed_meat.hpp
   arma_extend/fn_ccov.hpp
   arma_extend/glue_ccov_meat.hpp
   arma_extend/glue_ccov_proto.hpp
   arma_extend/op_ccov_meat.hpp
   arma_extend/op_ccov_proto.hpp
   arma_extend/operator_ostream.hpp
   arma_extend/promote_type.hpp
   arma_extend/traits.hpp
   arma_extend/typedef.hpp
   common.h
   compiler.h
   fortran.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at the parent cmake scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_library(base ${SOURCES})
