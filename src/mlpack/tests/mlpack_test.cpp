/**
 * @file tests/mlpack_test.cpp
 *
 * Simple file defining the name of the overall test for mlpack, and set up
 * global test fixture for each test. Each individual test is contained in
 * its own file.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_MODULE mlpackTest

#include <boost/version.hpp>

#if BOOST_VERSION >= 105900
  #include <boost/test/tree/visitor.hpp>
  #include <boost/test/tree/traverse.hpp>
#endif

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

/*
 * Class for traversing all Boost tests tree and building tree structure for
 * simple output.
 */
struct TestsVisitor : boost::unit_test::test_tree_visitor
{
  /*
   * Enter specified Boost unit test case.
   *
   * @param test Boost unit test case.
   */
  void visit(boost::unit_test::test_case const& test)
  {
    MLPACK_COUT_STREAM << std::string(indentations, ' ')
        << std::string(test.p_name) << "*" << std::endl;
  }

  /*
   * Enter specified Boost unit test suite.
   *
   * @param test Boost unit test suite.
   */
  bool test_suite_start(boost::unit_test::test_suite const& suite)
  {
    // For backward compatibility we omit the master suite.
    if (first)
    {
      first = false;
      return true;
    }

    MLPACK_COUT_STREAM << std::string(indentations, ' ')
        << std::string(suite.p_name) << "*" << std::endl;

    // Increase tab width (4 spaces).
    indentations += 4;

    return true;
  }

  /*
   * Leave specified Boost unit test suite.
   *
   * @param test Boost unit test case.
   */
  void test_suite_finish(boost::unit_test::test_suite const& /* suite */)
  {
    // Decrease tab width (4 spaces).
    indentations -= 4;
  }

  //! The indentation level.
  size_t indentations = 0;

  //! To keep track of where we are.
  bool first = true;
};

/**
 * Provide a global fixture for each test.
 *
 * A global fixture is expected to be implemented as a class where the class
 * constructor serves as a setup method and class destructor serves as teardown
 * method.
 *
 * By default, Log::objects should have their output redirected, otherwise
 * the UTF test output would be drowned out by Log::Debug and Log::Warn
 * messages.
 *
 * For more detailed test output, set the CMake flag TEST_VERBOSE=ON.
 */
struct GlobalFixture
{
  GlobalFixture()
  {
    #ifndef TEST_VERBOSE
      #ifdef DEBUG
        mlpack::Log::Debug.ignoreInput = true;
      #endif

      mlpack::Log::Info.ignoreInput = true;
      mlpack::Log::Warn.ignoreInput = true;
    #endif

    for (int i = 0; i < boost::unit_test::framework::master_test_suite().argc;
        ++i)
    {
      std::string argument(
          boost::unit_test::framework::master_test_suite().argv[i]);

      // Print Boost test hierarchy.
      if (argument == "--list_content")
      {
        TestsVisitor testsVisitor;
        traverse_test_tree(boost::unit_test::framework::master_test_suite(),
            testsVisitor);
        exit(0);
      }
    }
  }
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
