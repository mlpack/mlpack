# Simple rules for building mex file.  The _mex suffix is necessary to avoid
# target name conflicts, and the mex file must have a different name than the .m
# file.
add_library(range_search_mex SHARED
  range_search.cpp
)
target_link_libraries(range_search_mex
  mlpack
  ${LIBXML2_LIBRARIES}
)

# Installation rule.  Install both the mex and the MATLAB file.
install(TARGETS range_search_mex
  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
install(FILES
  range_search.m
  DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
