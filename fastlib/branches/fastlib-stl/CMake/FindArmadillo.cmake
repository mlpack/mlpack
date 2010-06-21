# Modified slightly for FASTLIB usage; no longer depends on other CMake code
# from the project this came from (http://usg.lofar.org)

# +-----------------------------------------------------------------------------+
# | $Id:: template_FindXX.cmake 1643 2008-06-14 10:19:20Z baehren             $ |
# +-----------------------------------------------------------------------------+
# |   Copyright (C) 2010                                                        |
# |   Lars B"ahren (bahren@astron.nl)                                           |
# |                                                                             |
# |   This program is free software; you can redistribute it and/or modify      |
# |   it under the terms of the GNU General Public License as published by      |
# |   the Free Software Foundation; either version 2 of the License, or         |
# |   (at your option) any later version.                                       |
# |                                                                             |
# |   This program is distributed in the hope that it will be useful,           |
# |   but WITHOUT ANY WARRANTY; without even the implied warranty of            |
# |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             |
# |   GNU General Public License for more details.                              |
# |                                                                             |
# |   You should have received a copy of the GNU General Public License         |
# |   along with this program; if not, write to the                             |
# |   Free Software Foundation, Inc.,                                           |
# |   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.                 |
# +-----------------------------------------------------------------------------+

# - Check for the presence of ARMADILLO
#
# The following variables are set when ARMADILLO is found:
#  HAVE_ARMADILLO       = Set to true, if all components of ARMADILLO have been
#                         found.
#  ARMADILLO_INCLUDES   = Include path for the header files of ARMADILLO
#  ARMADILLO_LIBRARIES  = Link these to use ARMADILLO
#  ARMADILLO_LFLAGS     = Linker flags (optional)

## -----------------------------------------------------------------------------
## Check for the header files

find_path (ARMADILLO_INCLUDES arma_ostream_proto.hpp
  PATHS 
  /usr/local/include
  /opt/include
  PATH_SUFFIXES armadillo armadillo_bits
  )

## -----------------------------------------------------------------------------
## Check for the library

find_library (ARMADILLO_LIBRARIES armadillo
  PATHS
  /usr/local/lib
  /opt/lib
  PATH_SUFFIXES
  )

## -----------------------------------------------------------------------------
## Actions taken when all components have been found

if (ARMADILLO_INCLUDES AND ARMADILLO_LIBRARIES)
  set (HAVE_ARMADILLO TRUE)
else (ARMADILLO_INCLUDES AND ARMADILLO_LIBRARIES)
  set (HAVE_ARMADILLO FALSE)
  if (NOT ARMADILLO_FIND_QUIETLY)
    if (NOT ARMADILLO_INCLUDES)
      message (STATUS "Unable to find ARMADILLO header files!")
    endif (NOT ARMADILLO_INCLUDES)
    if (NOT ARMADILLO_LIBRARIES)
      message (STATUS "Unable to find ARMADILLO library files!")
    endif (NOT ARMADILLO_LIBRARIES)
  endif (NOT ARMADILLO_FIND_QUIETLY)
endif (ARMADILLO_INCLUDES AND ARMADILLO_LIBRARIES)

if (HAVE_ARMADILLO)
  if (NOT ARMADILLO_FIND_QUIETLY)
    message (STATUS "Found components for ARMADILLO")
    message (STATUS "ARMADILLO_INCLUDES  = ${ARMADILLO_INCLUDES}")
    message (STATUS "ARMADILLO_LIBRARIES = ${ARMADILLO_LIBRARIES}")
  endif (NOT ARMADILLO_FIND_QUIETLY)
else (HAVE_ARMADILLO)
  if (ARMADILLO_FIND_REQUIRED)
    message (FATAL_ERROR "Could not find ARMADILLO!")
  endif (ARMADILLO_FIND_REQUIRED)
endif (HAVE_ARMADILLO)

## -----------------------------------------------------------------------------
## Mark advanced variables

mark_as_advanced (
  ARMADILLO_INCLUDES
  ARMADILLO_LIBRARIES
  )
