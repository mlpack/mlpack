# Define the files we need to compile
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  kernel_pca.hpp
  kernel_pca_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_subdirectory(kernel_rules)

add_executable(mlpack_kernel_pca
  kernel_pca_main.cpp
)
target_link_libraries(mlpack_kernel_pca
  mlpack
)
install(TARGETS mlpack_kernel_pca RUNTIME DESTINATION bin)
