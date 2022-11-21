# A very simple script to issue an error if the mlpack_test target is not
# defined.
message(FATAL_ERROR "To build the mlpack_test target, reconfigure CMake with the BUILD_TESTS option set to ON!  (i.e. `cmake -DBUILD_TESTS=ON ../`)")
