# add custom target distclean
# cleans and removes cmake generated files etc.
# Jan Woetzel 04/2003
#

# taken from http://cmake.org/pipermail/cmake/2003-June/003953.html
# hate at http://itk.org/Bug/view.php?id=6647
# yacked and brought out of 2003 by rcurtin

IF (UNIX)
  # since it's unix-specific we will use bash
  ADD_CUSTOM_TARGET (distclean @echo cleaning ${FASTLIB_SOURCE_DIR} for source distribution)

  ADD_CUSTOM_COMMAND(TARGET distclean
    COMMAND make ARGS clean
    COMMAND find ARGS ${FASTLIB_SOURCE_DIR} -iname CMakeCache.txt -delete
    COMMAND find ARGS ${FASTLIB_SOURCE_DIR} -iname cmake_install.cmake -delete
    COMMAND find ARGS ${FASTLIB_SOURCE_DIR} -iname Makefile -delete
    COMMAND find ARGS ${FASTLIB_SOURCE_DIR} -depth -type d -iname CMakeFiles -exec rm -rf {} \;
    COMMAND rm ARGS -rf bin lib include
    VERBATIM )
ENDIF(UNIX)
