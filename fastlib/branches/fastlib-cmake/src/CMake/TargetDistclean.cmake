# add custom target distclean
# cleans and removes cmake generated files etc.
# Jan Woetzel 04/2003
#

# taken from http://cmake.org/pipermail/cmake/2003-June/003953.html
# hate at http://itk.org/Bug/view.php?id=6647
# yacked and brought out of 2003 by rcurtin

IF (UNIX)
  ADD_CUSTOM_TARGET (distclean @echo cleaning for source distribution)
  SET(DISTCLEANED
   CMakeFiles
   CMakeCache.txt
   cmake_install.cmake
   Makefile
  )
  
  ADD_CUSTOM_COMMAND(
    DEPENDS clean
    COMMENT "distribution clean"
    COMMAND rm
    ARGS    -Rf CMakeFiles ${DISTCLEANED}
    TARGET  distclean
  )
ENDIF(UNIX)
