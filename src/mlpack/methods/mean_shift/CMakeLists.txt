# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  mean_shift.hpp
  mean_shift_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The main K-Means executable.
add_executable(mlpack_mean_shift
  mean_shift_main.cpp
)
target_link_libraries(mlpack_mean_shift
  mlpack
)
install(TARGETS mlpack_mean_shift RUNTIME DESTINATION bin)
