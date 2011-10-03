cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  # for mog_em
  mog_em.h
  mog_em.cc
  phi.h
  math_functions.h
  # for mog_l2e
  mog_l2e.h
  mog_l2e.cc
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# main executable, em
add_executable(mog_em
  mog_em_main.cc
)
# link dependencies of mog_em
target_link_libraries(mog_em
  mlpack
)

# main executable, l2e
add_executable(mog_l2e
  mog_l2e_main.cc
)
target_link_libraries(mog_l2e
  mlpack
)
