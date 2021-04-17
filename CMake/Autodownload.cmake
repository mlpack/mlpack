## This function auto-download mlpack dependencies.
## You need to pass the LINK to download from, the name of
## the dependency, and the name of the compressed package such as
## armadillo.tar.gz
## At each download, this module set a GENERIC_INCLUDE_DIR path,
## which mean that you need to set the main path for the include
## directories for each package.
## Note that, the package should be compressed only as .tar.gz

function(get_deps LINK DEPS_NAME PACKAGE)
  file(DOWNLOAD ${LINK}
         "${CMAKE_BINARY_DIR}/deps/${PACKAGE}"
          STATUS DOWNLOAD_STATUS_LIST LOG DOWNLOAD_LOG
          SHOW_PROGRESS)
    list(GET DOWNLOAD_STATUS_LIST 0 DOWNLOAD_STATUS)
    if (DOWNLOAD_STATUS EQUAL 0)
      execute_process(COMMAND ${CMAKE_COMMAND} -E
          tar xf "${CMAKE_BINARY_DIR}/deps/${PACKAGE}"
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/deps/")

      # Get the name of the directory.
      file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
          "${CMAKE_BINARY_DIR}/deps/${DEPS_NAME}*.*")
      if(${DEPS_NAME} MATCHES "boost")
        file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
            "${CMAKE_BINARY_DIR}/deps/${DEPS_NAME}*_*")
      elseif(${DEPS_NAME} MATCHES "stb")
        file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
            "${CMAKE_BINARY_DIR}/deps/${DEPS_NAME}")
      endif()
      # list(FILTER) is not available on 3.5 or older, but try to keep
      # configuring without filtering the list anyway 
      # (it works only the file is present as .tar.gz).
      if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.6.0")
        list(FILTER DIRECTORIES EXCLUDE REGEX ".*\.tar\.gz")
      endif ()
      list(LENGTH DIRECTORIES DIRECTORIES_LEN)
      message("Print directories: " ${DIRECTORIES})
      if (DIRECTORIES_LEN GREATER 0)
        list(GET DIRECTORIES 0 DEPENDENCY_DIR)
        set(GENERIC_INCLUDE_DIR "${CMAKE_BINARY_DIR}/deps/${DEPENDENCY_DIR}/include" CACHE INTERNAL "")

        if (${DEPS_NAME} MATCHES "boost")
          set(Boost_INCLUDE_DIR "${CMAKE_BINARY_DIR}/deps/${DEPENDENCY_DIR}/" CACHE INTERNAL "")

        elseif(${DEPS_NAME} MATCHES "OpenBLAS")
          if(NOT MSVC)
            execute_process(COMMAND make TARGET=ARMV8 BINARY=64 HOSTCC=gcc CC=${CMAKE_C_COMPILER} FC=${CMAKE_FORTRAN_COMPILER} NO_SHARED=1
                            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/${DEPENDENCY_DIR})
            file(GLOB OPENBLAS "${CMAKE_BINARY_DIR}/deps/${DEPENDENCY_DIR}/libopenblas.a")
            set(OPENBLAS_LIBRARIES ${OPENBLAS} CACHE INTERNAL "")
          endif()
        endif()

      else ()
        message(FATAL_ERROR 
                "Problem unpacking ${DEPS_NAME}! Expected only one directory ${DEPS_NAME};. Try removing the directory ${CMAKE_BINARY_DIR}/deps and reconfiguring.")
      endif ()
    else ()
      list(GET DOWNLOAD_STATUS_LIST 1 DOWNLOAD_ERROR)
      message(FATAL_ERROR
          "Could not download ${DEPS_NAME}! Error code ${DOWNLOAD_STATUS}: ${DOWNLOAD_ERROR}!  Error log: ${DOWNLOAD_LOG}")
    endif ()
endfunction()
