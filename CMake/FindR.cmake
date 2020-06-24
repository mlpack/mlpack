# -*- cmake -*-
#
# FindR.cmake: Try to find R
#
# (C) Copyright 2005-2012 EDF-EADS-Phimeca
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# @author dutka
# @date 2010-02-04 16:44:49 +0100 (Thu, 04 Feb 2010)
# Id Makefile.am 1473 2010-02-04 15:44:49Z dutka
#
#
# - Try to find R
# Once done this will define
#
# R_FOUND - System has R
# R_LIBRARIES - The libraries needed to use R
# R_DEFINITIONS - Compiler switches required for using R
# R_EXECUTABLE - The R interpreter


if ( R_EXECUTABLE AND R_LIBRARIES )
   # in cache already
   set( R_FIND_QUIETLY TRUE )
endif ( R_EXECUTABLE AND R_LIBRARIES )

#IF (NOT WIN32)
# # use pkg-config to get the directories and then use these values
# # in the FIND_PATH() and FIND_LIBRARY() calls
# FIND_PACKAGE(PkgConfig)
# PKG_CHECK_MODULES(PC_R R)
# SET(R_DEFINITIONS ${PC_R_CFLAGS_OTHER})
#ENDIF (NOT WIN32)

find_program ( R_EXECUTABLE
               NAMES R R.exe
               DOC "Path to the R command interpreter"
              )

get_filename_component ( _R_EXE_PATH ${R_EXECUTABLE} PATH )

if ( R_EXECUTABLE )
  execute_process ( COMMAND ${R_EXECUTABLE} RHOME
                    OUTPUT_VARIABLE _R_HOME
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )

  execute_process ( COMMAND ${R_EXECUTABLE} CMD config --cppflags
                    OUTPUT_VARIABLE R_CXX_FLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )

endif ( R_EXECUTABLE )

find_library ( R_LIBRARIES
  NAMES R
  HINTS
  ${_R_HOME}/lib
  ${_R_HOME}/lib/x86_64
)


set ( R_PACKAGES )
if ( R_EXECUTABLE )
  foreach ( _component ${R_FIND_COMPONENTS} )
    if ( NOT R_${_component}_FOUND )
    execute_process ( COMMAND echo "library(${_component})"
                      COMMAND ${R_EXECUTABLE} --no-save --silent --no-readline --slave
                      RESULT_VARIABLE _res
                      OUTPUT_VARIABLE _trashout
                      ERROR_VARIABLE _trasherr
                    )
    if ( NOT _res )
      message ( STATUS "Looking for R package ${_component} - found" )
      set ( R_${_component}_FOUND 1 CACHE INTERNAL "True if R package ${_component} is here" )
    else ( NOT _res )
      message ( STATUS "Looking for R package ${_component} - not found" )
      set ( R_${_component}_FOUND 0 CACHE INTERNAL "True if R package ${_component} is here" )
    endif ( NOT _res )
    list ( APPEND R_PACKAGES R_${_component}_FOUND )
    endif ( NOT R_${_component}_FOUND )
  endforeach ( _component )
endif ( R_EXECUTABLE )

include ( FindPackageHandleStandardArgs )

# handle the QUIETLY and REQUIRED arguments and set R_FOUND to TRUE if
# all listed variables are TRUE
find_package_handle_standard_args ( R DEFAULT_MSG R_EXECUTABLE R_LIBRARIES R_CXX_FLAGS ${R_PACKAGES} )

mark_as_advanced ( R_EXECUTABLE R_LIBRARIES R_CXX_FLAGS ${R_PACKAGES} )
