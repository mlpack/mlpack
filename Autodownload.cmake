function(get_deps LINK DEPS_NAME PACKAGE)

  message("Print link to download: " ${LINK})
  message("Print deps name: " ${DEPS_NAME})
  message("Print deps name: " ${PACKAGE})

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
            "${CMAKE_BINARY_DIR}/deps/${PACKAGE}.tar")
        # list(FILTER) is not available on 3.5 or older, but try to keep
        # configuring without filtering the list anyway (it might work if only
        # the file ensmallen-latest.tar.gz is present.
        if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.6.0")
          list(FILTER DIRECTORIES EXCLUDE REGEX ".*\.tar\.xz")
        endif ()
        list(LENGTH DIRECTORIES DIRECTORIES_LEN)
        if (DIRECTORIES_LEN EQUAL 1)
          if (${DEPS_NAME} EQUAL "armadillo")
            list(GET DIRECTORIES 0 ARMADILLO_INCLUDE_DIR)
            set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
                "${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/include")
            message(STATUS
                "Successfully downloaded ${DEPS_NAME} into ${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/")

            # Now we have to also ensure these header files get installed.
            install(DIRECTORY "${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/include/armadillo_bits/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/armadillo_bits")
            install(FILES "${CMAKE_BINARY_DIR}/deps/${ARMADILLO_INCLUDE_DIR}/include/armadillo" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

          elseif (${DEPS_NAME} MATCHES "ensmallen")
            list(GET DIRECTORIES 0 ENSMALLEN_INCLUDE_DIR)
            set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
               "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include")
            message(STATUS
               "Successfully downloaded ${DEPS_NAME}$ into ${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/")

            # Now we have to also ensure these header files get installed.
            install(DIRECTORY "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include/ensmallen_bits/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ensmallen_bits")
            install(FILES "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include/ensmallen.hpp" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

          elseif (${DEPS_NAME} EQUAL "cereal")
            list(GET DIRECTORIES 0 CEREAL_INCLUDE_DIR)
            set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
                "${CMAKE_BINARY_DIR}/deps/${CEREAL_INCLUDE_DIR}/include")
            message(STATUS
                "Successfully downloaded ${DEPS_NAME} into ${CMAKE_BINARY_DIR}/deps/${CEREAL_INCLUDE_DIR}/")

            # Now we have to also ensure these header files get installed.
            install(DIRECTORY "${CMAKE_BINARY_DIR}/deps/${CEREAL_INCLUDE_DIR}/include/cereal" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cereal")

          elseif(${DEPS_NAME} EQUAL "boost")
            list(GET DIRECTORIES 0 Boost_INCLUDE_DIR)
            set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
                "${CMAKE_BINARY_DIR}/deps/${Boost_INCLUDE_DIR}")
            message(STATUS
                "Successfully downloaded ${DEPS_NAME} into ${CMAKE_BINARY_DIR}/deps/${Boost_INCLUDE_DIR}/")

          endif()

        else ()
          message(FATAL_ERROR 
                  "Problem unpacking ${DEPS_NAME}! Expected only one directory ${DEPS_NAME}-;. Try removing the directory ${CMAKE_BINARY_DIR}/deps and reconfiguring.")
        endif ()
      endif ()
      if (${DEPS_NAME} EQUAL "stb")
          file(MAKE_DIRECTORY ${stb})
          if(PACKAGE EQUAL "stb_image.h")
            file(RENAME "stb_image.h" "stb/stb_image.h")
            check_hash (http://mlpack.org/files/stb/hash.md5
                "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image.h"
                HASH_CHECK_FAIL)
          elseif(PACKAGE EQUAL "stb_image_write.h")
            file(RENAME "stb_image_write.h" "stb/stb_image_write.h")
            check_hash (http://mlpack.org/files/stb/hash.md5
                "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image_write.h"
                HASH_CHECK_FAIL)
          endif()
          if (HASH_CHECK_FAIL EQUAL 0)
            set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
                "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/")
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
    else ()
      list(GET DOWNLOAD_STATUS_LIST 1 DOWNLOAD_ERROR)
      message(FATAL_ERROR
          "Could not download armadillo! Error code ${DOWNLOAD_STATUS}: ${DOWNLOAD_ERROR}!  Error log: ${DOWNLOAD_LOG}")
    endif ()
endfunction()
