# Simple rules for building mex file.  The _mex suffix is necessary to avoid
# target name conflicts, and the mex file must have a different name than the .m
# file.
add_library(kmeans_mex SHARED
  kmeans.cpp
)
target_link_libraries(kmeans_mex
  mlpack
  ${LIBXML2_LIBRARIES}
)

# Installation rule.  Install both the mex and the MATLAB file.
install(TARGETS kmeans_mex
  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
install(FILES
  kmeans.m
  DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
