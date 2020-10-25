/**
 * @file tests/python_binding_test.cpp
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

#include "catch.hpp"

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

/**
 * Ensure that we can construct a PyOption object, and that it will add itself
 * to the IO instance.
 */
TEST_CASE("PyOptionTest", "[PythonBindingsTest]")
{
  IO::ClearSettings();
  programName = "test";
  PyOption<double> po1(0.0, "test", "test2", "t", "double", false, true, false);

  // Now check that it's in IO.
  IO::RestoreSettings(programName);
  REQUIRE(IO::Parameters().count("test") > 0);
  REQUIRE(IO::Aliases().count('t') > 0);
  REQUIRE(IO::Parameters()["test"].desc == "test2");
  REQUIRE(IO::Parameters()["test"].name == "test");
  REQUIRE(IO::Parameters()["test"].alias == 't');
  REQUIRE(IO::Parameters()["test"].noTranspose == false);
  REQUIRE(IO::Parameters()["test"].required == false);
  REQUIRE(IO::Parameters()["test"].input == true);
  REQUIRE(IO::Parameters()["test"].cppType == "double");

  PyOption<arma::mat> po2(arma::mat(), "mat", "mat2", "m", "arma::mat", true,
      true, true);

  // Now check that it's in IO.
  IO::RestoreSettings(programName);
  REQUIRE(IO::Parameters().count("mat") > 0);
  REQUIRE(IO::Aliases().count('m') > 0);
  REQUIRE(IO::Parameters()["mat"].desc == "mat2");
  REQUIRE(IO::Parameters()["mat"].name == "mat");
  REQUIRE(IO::Parameters()["mat"].alias == 'm');
  REQUIRE(IO::Parameters()["mat"].noTranspose == true);
  REQUIRE(IO::Parameters()["mat"].required == true);
  REQUIRE(IO::Parameters()["mat"].input == true);
  REQUIRE(IO::Parameters()["mat"].cppType == "arma::mat");

  IO::ClearSettings();
}

/**
 * Make sure GetParam() works.
 */
TEST_CASE("PyGetParamDoubleTest", "[PythonBindingsTest]")
{
  util::ParamData d;
  double x = 5.0;
  d.value = boost::any(x);

  double* output = NULL;
  GetParam<double>(d, (void*) NULL, (void*) &output);

  REQUIRE(*output == 5.0);
}

TEST_CASE("GetParamMatTest", "[PythonBindingsTest]")
{
  util::ParamData d;
  arma::mat m(5, 5, arma::fill::ones);
  d.value = boost::any(m);

  arma::mat* output = NULL;
  GetParam<arma::mat>(d, (void*) NULL, (void*) &output);

  REQUIRE(output->n_rows == 5);
  REQUIRE(output->n_cols == 5);
  for (size_t i = 0; i < 25; ++i)
    REQUIRE((*output)[i] == 1.0);
}

/**
 * All of the other functions are implicitly tested simply by compilation.
 */
