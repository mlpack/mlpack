# Simple rules for building mex file.  The _mex suffix is necessary to avoid
# target name conflicts, and the mex file must have a different name than the .m
# file.
add_library(gmm_mex SHARED
  gmm.cpp
)
target_link_libraries(gmm_mex
  mlpack
  ${LIBXML2_LIBRARIES}
)

# Installation rule.  Install both the mex and the MATLAB file.
install(TARGETS gmm_mex
  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
install(FILES
  gmm.m
  DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
