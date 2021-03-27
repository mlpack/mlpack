function(get_deps LINK DEPS_NAME PACKAGE)

  file(DOWNLOAD ${LINK}
         "${CMAKE_BINARY_DIR}/deps/${PACKAGE}"
          STATUS DOWNLOAD_STATUS_LIST LOG DOWNLOAD_LOG
          SHOW_PROGRESS)
    list(GET DOWNLOAD_STATUS_LIST 0 DOWNLOAD_STATUS)
    if (DOWNLOAD_STATUS EQUAL 0)
      if (NOT ${DEPS_NAME} MATCHES "stb")
        execute_process(COMMAND ${CMAKE_COMMAND} -E
            tar xf "${CMAKE_BINARY_DIR}/deps/${PACKAGE}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/deps/")

        # Get the name of the directory.
        file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
            "${CMAKE_BINARY_DIR}/deps/${DEPS_NAME}*.*")
        if(${DEPS_NAME} MATCHES "boost")
          file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
              "${CMAKE_BINARY_DIR}/deps/${DEPS_NAME}*_*")
        endif()
        # list(FILTER) is not available on 3.5 or older, but try to keep
        # configuring without filtering the list anyway (it might work if only
        # the file ensmallen-latest.tar.gz is present.
        if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.6.0")
          list(FILTER DIRECTORIES EXCLUDE REGEX ".*\.tar\.gz")
          list(FILTER DIRECTORIES EXCLUDE REGEX ".*\.tar\.xz")
        endif ()
        list(LENGTH DIRECTORIES DIRECTORIES_LEN)
        message("Print directories: " ${DIRECTORIES})
        if (DIRECTORIES_LEN GREATER 0)
          if (${DEPS_NAME} MATCHES "armadillo")
            list(GET DIRECTORIES 0 ARMADILLO_DIR)
            set(ARMADILLO_INCLUDE_DIR "${CMAKE_BINARY_DIR}/deps/${ARMADILLO_DIR}/include" CACHE INTERNAL "")

          elseif (${DEPS_NAME} MATCHES "ensmallen")
            list(GET DIRECTORIES 0 ENSMALLEN_DIR)
            set(ENSMALLEN_INCLUDE_DIR "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_DIR}/include" CACHE INTERNAL "")

          elseif (${DEPS_NAME} MATCHES "cereal")
            list(GET DIRECTORIES 0 CEREAL_DIR)
            set(CEREAL_INCLUDE_DIR "${CMAKE_BINARY_DIR}/deps/${CEREAL_DIR}/include" CACHE INTERNAL "")

          elseif(${DEPS_NAME} MATCHES "boost")
            list(GET DIRECTORIES 0 Boost_DIR)
            set(Boost_INCLUDE_DIR "${CMAKE_BINARY_DIR}/deps/${Boost_DIR}/" CACHE INTERNAL "")

          endif()

        else ()
          message(FATAL_ERROR 
                  "Problem unpacking ${DEPS_NAME}! Expected only one directory ${DEPS_NAME};. Try removing the directory ${CMAKE_BINARY_DIR}/deps and reconfiguring.")
        endif ()
      endif ()
      if (${DEPS_NAME} MATCHES "stb")
        set(STB_DIR "stb")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/deps/stb")
        if(PACKAGE MATCHES "stb_image.h")
          execute_process(COMMAND mv "deps/stb_image.h" "deps/stb")
        elseif(PACKAGE MATCHES "stb_image_write.h")
          execute_process(COMMAND mv "deps/stb_image_write.h" "deps/stb")
        endif()
        if(EXISTS "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image.h" AND EXISTS "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image_write.h")
          check_hash (http://mlpack.org/files/stb/hash.md5
              "${CMAKE_BINARY_DIR}/deps/${STB_DIR}"
              HASH_CHECK_FAIL)
          if (HASH_CHECK_FAIL EQUAL 0)
            set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
                "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/" CACHE INTERNAL "")
            message(STATUS
                "Successfully downloaded stb into ${CMAKE_BINARY_DIR}/deps/${STB_DIR}/")
            # Now we have to also ensure these header files get installed.
            install(FILES "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
            install(FILES "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image_write.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
            add_definitions(-DHAS_STB)
            set(STB_AVAILABLE "1")
          else ()
            message(WARNING
                "stb/stb_image.h is not installed. Image utilities will not be available!")
          endif()
        endif()
      endif()
    else ()
      list(GET DOWNLOAD_STATUS_LIST 1 DOWNLOAD_ERROR)
      message(FATAL_ERROR
          "Could not download armadillo! Error code ${DOWNLOAD_STATUS}: ${DOWNLOAD_ERROR}!  Error log: ${DOWNLOAD_LOG}")
    endif ()
endfunction()
