# Simple rules for building mex file.  The _mex suffix is necessary to avoid
# target name conflicts, and the mex file must have a different name than the .m
# file.
add_library(nmf_mex SHARED
  nmf.cpp
)
target_link_libraries(nmf_mex
  mlpack
  ${LIBXML2_LIBRARIES}
)

# Installation rule.  Install both the mex and the MATLAB file.
install(TARGETS nmf_mex
  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
install(FILES
  nmf.m
  DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
