/**
 * @file mlpack_test.cpp
 *
 * Simple file defining the name of the overall test for MLPACK.  Each
 * individual test is contained in its own file.
 */
#define BOOST_TEST_MODULE MLPACKTest

#include <boost/version.hpp>

// We only need to do this for old Boost versions.
#if BOOST_VERSION < 103600
  #define BOOST_AUTO_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"
