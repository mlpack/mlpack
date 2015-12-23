# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  fastmks.hpp
  fastmks_impl.hpp
  fastmks_model.hpp
  fastmks_model_impl.hpp
  fastmks_model.cpp
  fastmks_rules.hpp
  fastmks_rules_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(mlpack_fastmks
  fastmks_main.cpp
)
target_link_libraries(mlpack_fastmks
  mlpack
)
install(TARGETS mlpack_fastmks RUNTIME DESTINATION bin)
