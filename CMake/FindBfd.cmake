# - Try to find libbfd
# Once done this will define
#
#  LIBBFD_FOUND - system has libbfd
#  LIBBFD_INCLUDE_DIRS - the libbfd include directory
#  LIBBFD_LIBRARIES - Link these to use libbfd
#  LIBBFD_DEFINITIONS - Compiler switches required for using libbfd
#
# Based on:
#
#  Copyright (c) 2008 Bernhard Walle <bernhard.walle@gmx.de>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#


if (LIBBFD_LIBRARIES AND LIBBFD_INCLUDE_DIRS)
  set (LIBBFD_FIND_QUIETLY TRUE)
endif ()

find_path (LIBBFD_INCLUDE_DIRS
    NAMES
      bfd.h
      dis-asm.h
    PATHS
      /usr/include
      /usr/local/include
      /opt/local/include
      /opt/include
      ENV CPATH)

# Ugly, yes ugly...
find_library (LIBBFD_BFD_LIBRARY
    NAMES
      bfd
    PATHS
      /usr/lib
      /usr/lib64
      /usr/local/lib
      /usr/local/lib64
      /usr/include
      /opt/local/lib
      /opt/usr/lib64
      ENV LIBRARY_PATH
      ENV LD_LIBRARY_PATH)

#find_library (LIBBFD_IBERTY_LIBRARY
#    NAMES
#      iberty
#    PATHS
#      /usr/lib
#      /usr/lib64
#      /usr/local/lib
#      /usr/local/lib64
#      /usr/include
#      /opt/local/lib
#      /opt/usr/lib64
#      ENV LIBRARY_PATH
#      ENV LD_LIBRARY_PATH)

#find_library (LIBBFD_OPCODES_LIBRARY
#    NAMES
#      opcodes
#    PATHS
#      /usr/lib
#      /usr/lib64
#      /usr/local/lib
#      /usr/local/lib64
#      /usr/include
#      /opt/local/lib
#      /opt/usr/lib64
#      ENV LIBRARY_PATH
#      ENV LD_LIBRARY_PATH)


include (FindPackageHandleStandardArgs)


# handle the QUIETLY and REQUIRED arguments and set LIBBFD_FOUND to TRUE if all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBBFD DEFAULT_MSG
    LIBBFD_BFD_LIBRARY
#    LIBBFD_IBERTY_LIBRARY
#    LIBBFD_OPCODES_LIBRARY
    LIBBFD_INCLUDE_DIRS)

set(LIBBFD_LIBRARIES "${LIBBFD_BFD_LIBRARY}")
mark_as_advanced(LIBBFD_INCLUDE_DIRS LIBBFD_LIBRARIES)