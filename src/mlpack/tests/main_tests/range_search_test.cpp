/**
 * @file range_search_test.cpp
 * @author Niteya Shah
 *
 * Test mlpackMain() of range_search_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "RangeSearchMain";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/range_search/range_search_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct RangeSearchTestFixture
{
 public:

  RangeSearchTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }
  ~RangeSearchTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
BOOST_FIXTURE_TEST_SUITE(RangeSearchMainTest, RangeSearchTestFixture);
BOOST_AUTO_TEST_CASE(SyntheticRangeSearch)
{
  mlpackMain();
}
BOOST_AUTO_TEST_SUITE_END();
