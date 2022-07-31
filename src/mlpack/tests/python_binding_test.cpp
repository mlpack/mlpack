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

using namespace mlpack;
using namespace mlpack::bindings;
using namespace mlpack::bindings::python;

// If multiple binding types are used in the same test file, we may get the
// wrong function map.  These functions are utilities to ensure that for these
// tests, the function maps are accurate.

template<typename N>
void AddPyMapFunctions(util::Params& p)
{
  p.functionMap[TYPENAME(N)]["GetParam"] = &python::GetParam<N>;
  p.functionMap[TYPENAME(N)]["GetPrintableParam"] =
      &python::GetPrintableParam<N>;
  p.functionMap[TYPENAME(N)]["DefaultParam"] = &python::DefaultParam<N>;
  p.functionMap[TYPENAME(N)]["PrintClassDefn"] = &python::PrintClassDefn<N>;
  p.functionMap[TYPENAME(N)]["PrintDefn"] = &python::PrintDefn<N>;
  p.functionMap[TYPENAME(N)]["PrintDoc"] = &python::PrintDoc<N>;
  p.functionMap[TYPENAME(N)]["PrintOutputProcessing"] =
      &python::PrintOutputProcessing<N>;
  p.functionMap[TYPENAME(N)]["PrintInputProcessing"] =
      &python::PrintInputProcessing<N>;
  p.functionMap[TYPENAME(N)]["ImportDecl"] = &python::ImportDecl<N>;
}

/**
 * Ensure that we can construct a PyOption object, and that it will add itself
 * to the IO instance.
 */
TEST_CASE("PyOptionTest", "[PythonBindingsTest]")
{
  PyOption<double> po1(0.0, "test", "test2", "t", "double", false, true, false,
      "PyOptionTest");

  // Now check that it's in IO.
  util::Params p = IO::Parameters("PyOptionTest");
  p.functionMap.clear();
  AddPyMapFunctions<double>(p);
  REQUIRE(p.Parameters().count("test") > 0);
  REQUIRE(p.Aliases().count('t') > 0);
  REQUIRE(p.Parameters()["test"].desc == "test2");
  REQUIRE(p.Parameters()["test"].name == "test");
  REQUIRE(p.Parameters()["test"].alias == 't');
  REQUIRE(p.Parameters()["test"].noTranspose == false);
  REQUIRE(p.Parameters()["test"].required == false);
  REQUIRE(p.Parameters()["test"].input == true);
  REQUIRE(p.Parameters()["test"].cppType == "double");

  PyOption<arma::mat> po2(arma::mat(), "mat", "mat2", "m", "arma::mat", true,
      true, true, "PyOptionTest");

  // Now check that it's in IO.
  p = IO::Parameters("PyOptionTest");
  p.functionMap.clear();
  AddPyMapFunctions<arma::mat>(p);
  REQUIRE(p.Parameters().count("mat") > 0);
  REQUIRE(p.Aliases().count('m') > 0);
  REQUIRE(p.Parameters()["mat"].desc == "mat2");
  REQUIRE(p.Parameters()["mat"].name == "mat");
  REQUIRE(p.Parameters()["mat"].alias == 'm');
  REQUIRE(p.Parameters()["mat"].noTranspose == true);
  REQUIRE(p.Parameters()["mat"].required == true);
  REQUIRE(p.Parameters()["mat"].input == true);
  REQUIRE(p.Parameters()["mat"].cppType == "arma::mat");
}

/**
 * Make sure GetParam() works.
 */
TEST_CASE("PyGetParamDoubleTest", "[PythonBindingsTest]")
{
  util::ParamData d;
  double x = 5.0;
  d.value = x;

  double* output = NULL;
  GetParam<double>(d, (void*) NULL, (void*) &output);

  REQUIRE(*output == 5.0);
}

TEST_CASE("GetParamMatTest", "[PythonBindingsTest]")
{
  util::ParamData d;
  arma::mat m(5, 5, arma::fill::ones);
  d.value = m;

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
