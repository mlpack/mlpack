add_subdirectory(metrics)

# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  cv_base.hpp
  cv_base_impl.hpp
  k_fold_cv.hpp
  k_fold_cv_impl.hpp
  meta_info_extractor.hpp
  simple_cv.hpp
  simple_cv_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
