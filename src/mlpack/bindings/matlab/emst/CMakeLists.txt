# Simple rules for building mex file.  The _mex suffix is necessary to avoid
# target name conflicts.
add_library(emst_mex SHARED
  emst.cpp
)
target_link_libraries(emst_mex
  mlpack
  ${LIBXML2_LIBRARIES}
)

# Change the name of the output library.
set_target_properties(emst_mex
  PROPERTIES OUTPUT_NAME emst
)

# Installation rule.  Install both the mex and the MATLAB file.
install(TARGETS emst_mex
  LIBRARY DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
install(FILES
  emst.m
  DESTINATION "${MATLAB_TOOLBOX_DIR}/mlpack/"
)
