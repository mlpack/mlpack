function(GET_DEPS LINK DEPS_NAME)

  message("Print link to download" LINK)
  message("Print deps name" DEPS_NAME)

  if(DEPS_NAME EQUAL "ensmallen")
    set(PACKAGE EQUAL "ensmallen-latest")
  elseif(DEPS_NAME EQUAL "cereal")
    set(PACKAGE EQUAL "cereal")
  elseif(DEPS_NAME EQUAL "armadillo")
    set(PACKAGE EQUAL "armadillo-9.900.1")
  elseif(DEPS_NAME EQUAL "boost")
    set(PACKAGE EQUAL "boost")
  endif()

  file(DOWNLOAD LINK
         "${CMAKE_BINARY_DIR}/deps/${PACKAGE}.tar.xz"
          STATUS DOWNLOAD_STATUS_LIST LOG DOWNLOAD_LOG
          SHOW_PROGRESS)
    list(GET DOWNLOAD_STATUS_LIST 0 DOWNLOAD_STATUS)
    if (DOWNLOAD_STATUS EQUAL 0)
      execute_process(COMMAND ${CMAKE_COMMAND} -E
          tar xf "${CMAKE_BINARY_DIR}/deps/${PACKAGE}.tar.xz"
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/deps/")

      # Get the name of the directory.
      file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
          "${CMAKE_BINARY_DIR}/deps/armadillo*.*")
      # list(FILTER) is not available on 3.5 or older, but try to keep
      # configuring without filtering the list anyway (it might work if only
      # the file cereal-latest.tar.gz is present.
      if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.6.0")
        list(FILTER DIRECTORIES EXCLUDE REGEX ".*\.tar\.xz")
      endif ()
      list(LENGTH DIRECTORIES DIRECTORIES_LEN)
      if (DIRECTORIES_LEN EQUAL 1)
      
        if(DEPS_NAME EQUAL "armadillo")
          list(GET DIRECTORIES 0 ARMADILLO_INCLUDE_DIR)
          set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
              "${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/include")
          message(STATUS
              "Successfully downloaded armadillo into ${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/")

          # Now we have to also ensure these header files get installed.
          install(DIRECTORY "${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/include/armadillo_bits/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/armadillo_bits")
          install(FILES "${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/include/armadillo" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

        elseif(DEPS_NAME EQUAL "ensmallen")
          list(GET DIRECTORIES 0 ENSMALLEN_INCLUDE_DIR)
          set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
             "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include")
          message(STATUS
             "Successfully downloaded ensmallen into ${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/")

          # Now we have to also ensure these header files get installed.
          install(DIRECTORY "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include/ensmallen_bits/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ensmallen_bits")
          install(FILES "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include/ensmallen.hpp" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
        endif()

        elseif(DEPS_NAME EQUAL "cereal")

        endif()

        else ()
          message(FATAL_ERROR 
                  "Problem unpacking ${DEPS_NAME}! Expected only one directory ${DEPS_NAME}-;. Try removing the directory ${CMAKE_BINARY_DIR}/deps and reconfiguring.")
        endif ()
    else ()
      list(GET DOWNLOAD_STATUS_LIST 1 DOWNLOAD_ERROR)
      message(FATAL_ERROR
          "Could not download armadillo! Error code ${DOWNLOAD_STATUS}: ${DOWNLOAD_ERROR}!  Error log: ${DOWNLOAD_LOG}")
    endif ()

endfunction()
