# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  binary_layer.hpp
  binary_layer_impl.hpp
  binary_rbm.hpp 
  gan.hpp
  rbm.hpp
  rbm_impl.hpp
  spike_slab_layer.hpp
  spike_slab_layer_impl.hpp
  ssRBM.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
