# Simple rules for building mex file.  The _mex suffix is necessary to avoid
# target name conflicts, and the mex file must have a different name than the .m
# file.
add_library(kernel_pca_mex SHARED
  kernel_pca.cpp
)
target_link_libraries(kernel_pca_mex
  mlpack
  ${LIBXML2_LIBRARIES}
)

# Installation rule.  Install both the mex and the MATLAB file.
install(TARGETS kernel_pca_mex
  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
install(FILES
  kernel_pca.m
  DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
