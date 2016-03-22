# Recurse into individual binding subdirectories, if we are supposed to.
if(MATLAB_BINDINGS)
  add_subdirectory(matlab)
endif()
