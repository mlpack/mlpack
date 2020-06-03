/**
 * @file python_binding_test.cpp
 * @author Ryan Curtin
 *
 * Test the components of the Python bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/bindings/python/py_option.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

namespace mlpack {
namespace bindings {
namespace python {

std::string programName; // Needed for linking.

} // namespace python
} // namespace bindings
} // namespace mlpack

using namespace mlpack;
using namespace mlpack::bindings;
using namespace mlpack::bindings::python;

BOOST_AUTO_TEST_SUITE(PythonBindingsTest);

/**
 * Ensure that we can construct a PyOption object, and that it will add itself
 * to the CMD instance.
 */
BOOST_AUTO_TEST_CASE(PyOptionTest)
{
  CMD::ClearSettings();
  programName = "test";
  PyOption<double> po1(0.0, "test", "test2", "t", "double", false, true, false);

  // Now check that it's in CMD.
  CMD::RestoreSettings(programName);
  BOOST_REQUIRE_GT(CMD::Parameters().count("test"), 0);
  BOOST_REQUIRE_GT(CMD::Aliases().count('t'), 0);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["test"].desc, "test2");
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["test"].name, "test");
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["test"].alias, 't');
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["test"].noTranspose, false);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["test"].required, false);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["test"].input, true);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["test"].cppType, "double");

  PyOption<arma::mat> po2(arma::mat(), "mat", "mat2", "m", "arma::mat", true,
      true, true);

  // Now check that it's in CMD.
  CMD::RestoreSettings(programName);
  BOOST_REQUIRE_GT(CMD::Parameters().count("mat"), 0);
  BOOST_REQUIRE_GT(CMD::Aliases().count('m'), 0);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["mat"].desc, "mat2");
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["mat"].name, "mat");
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["mat"].alias, 'm');
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["mat"].noTranspose, true);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["mat"].required, true);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["mat"].input, true);
  BOOST_REQUIRE_EQUAL(CMD::Parameters()["mat"].cppType, "arma::mat");

  CMD::ClearSettings();
}

/**
 * Make sure GetParam() works.
 */
BOOST_AUTO_TEST_CASE(GetParamDoubleTest)
{
  util::ParamData d;
  double x = 5.0;
  d.value = boost::any(x);

  double* output = NULL;
  GetParam<double>(d, (void*) NULL, (void*) &output);

  BOOST_REQUIRE_EQUAL(*output, 5.0);
}

BOOST_AUTO_TEST_CASE(GetParamMatTest)
{
  util::ParamData d;
  arma::mat m(5, 5, arma::fill::ones);
  d.value = boost::any(m);

  arma::mat* output = NULL;
  GetParam<arma::mat>(d, (void*) NULL, (void*) &output);

  BOOST_REQUIRE_EQUAL(output->n_rows, 5);
  BOOST_REQUIRE_EQUAL(output->n_cols, 5);
  for (size_t i = 0; i < 25; ++i)
    BOOST_REQUIRE_EQUAL((*output)[i], 1.0);
}

/**
 * All of the other functions are implicitly tested simply by compilation.
 */

BOOST_AUTO_TEST_SUITE_END();
