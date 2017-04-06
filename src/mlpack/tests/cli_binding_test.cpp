/**
 * @file cli_binding_test.cpp
 * @author Ryan Curtin
 *
 * Test the components of the CLI bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/bindings/cli/cli_option.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::bindings;
using namespace mlpack::bindings::cli;
using namespace mlpack::kernel;

BOOST_AUTO_TEST_SUITE(CLIBindingTest);

/**
 * Ensure that we can construct a CLIOption object, and that it will add itself
 * to the CLI instance.
 */
BOOST_AUTO_TEST_CASE(CLIOptionTest)
{
  CLIOption<double> co1(0.0, "test", "test2", "t", "double", false, true,
      false);

  // Now check that it's in CLI.
  BOOST_REQUIRE_GT(CLI::Parameters().count("test"), 0);
  BOOST_REQUIRE_GT(CLI::Aliases().count('t'), 0);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["test"].desc, "test2");
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["test"].name, "test");
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["test"].alias, 't');
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["test"].noTranspose, false);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["test"].required, false);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["test"].input, true);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["test"].cppType, "double");

  CLIOption<arma::mat> co2(arma::mat(), "mat", "mat2", "m", "arma::mat", true,
      true, true);

  // Now check that it's in CLI.
  BOOST_REQUIRE_GT(CLI::Parameters().count("mat"), 0);
  BOOST_REQUIRE_GT(CLI::Aliases().count('m'), 0);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["mat"].desc, "mat2");
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["mat"].name, "mat");
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["mat"].alias, 'm');
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["mat"].noTranspose, true);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["mat"].required, true);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["mat"].input, true);
  BOOST_REQUIRE_EQUAL(CLI::Parameters()["mat"].cppType, "arma::mat");

  CLI::ClearSettings();
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
  GetParam<double>((const util::ParamData&) d, (const void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL(*output, 5.0);
}

BOOST_AUTO_TEST_CASE(GetParamLoadedMatTest)
{
  util::ParamData d;
  // Create value.
  string filename = "hello.csv";
  arma::mat m(5, 5, arma::fill::ones);
  tuple<arma::mat, string> tuple = make_tuple(m, filename);
  d.value = boost::any(tuple);
  // Mark it as already loaded.
  d.input = true;
  d.loaded = true;

  arma::mat* output = NULL;
  GetParam<arma::mat>((const util::ParamData&) d, (void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL(output->n_rows, 5);
  BOOST_REQUIRE_EQUAL(output->n_cols, 5);
  for (size_t i = 0; i < 25; ++i)
    BOOST_REQUIRE_EQUAL((*output)[i], 1.0);
}

BOOST_AUTO_TEST_CASE(GetParamUnloadedMatTest)
{
  util::ParamData d;
  // Create value.
  string filename = "test.csv";
  arma::mat test(5, 5, arma::fill::ones);
  data::Save("test.csv", test);
  arma::mat m;
  tuple<arma::mat, string> tuple = make_tuple(m, filename);
  d.value = boost::any(tuple);
  // Make sure it is not loaded yet.
  d.input = true;
  d.loaded = false;

  // Now getting the parameter should load it.
  arma::mat* output = NULL;
  GetParam<arma::mat>((const util::ParamData&) d, (void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL(output->n_rows, 5);
  BOOST_REQUIRE_EQUAL(output->n_cols, 5);
  for (size_t i = 0; i < 25; ++i)
    BOOST_REQUIRE_EQUAL((*output)[i], 1.0);

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(GetParamUmatTest)
{
  util::ParamData d;
  // Create value.
  string filename = "hello.csv";
  arma::Mat<size_t> m(5, 5, arma::fill::ones);
  tuple<arma::Mat<size_t>, string> tuple = make_tuple(m, filename);
  d.value = boost::any(tuple);
  // Mark it as already loaded.
  d.input = true;
  d.loaded = true;

  arma::Mat<size_t>* output = NULL;
  GetParam<arma::Mat<size_t>>((const util::ParamData&) d, (void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL(output->n_rows, 5);
  BOOST_REQUIRE_EQUAL(output->n_cols, 5);
  for (size_t i = 0; i < 25; ++i)
    BOOST_REQUIRE_EQUAL((*output)[i], 1.0);
}

BOOST_AUTO_TEST_CASE(GetParamUnloadedUmatTest)
{
  util::ParamData d;
  // Create value.
  string filename = "test.csv";
  arma::Mat<size_t> test(5, 5, arma::fill::ones);
  data::Save("test.csv", test);
  arma::Mat<size_t> m;
  tuple<arma::Mat<size_t>, string> tuple = make_tuple(m, filename);
  d.value = boost::any(tuple);
  // Make sure it is not loaded yet.
  d.input = true;
  d.loaded = false;

  // Now getting the parameter should load it.
  arma::Mat<size_t>* output = NULL;
  GetParam<arma::Mat<size_t>>((const util::ParamData&) d, (void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL(output->n_rows, 5);
  BOOST_REQUIRE_EQUAL(output->n_cols, 5);
  for (size_t i = 0; i < 25; ++i)
    BOOST_REQUIRE_EQUAL((*output)[i], 1.0);

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(GetParamDatasetInfoMatTest)
{
  util::ParamData d;

  // Create value.
  string filename = "test.csv";

  fstream f;
  f.open("test.csv", fstream::out);
  f << "1, 2, hello" << endl;
  f << "3, 4, goodbye" << endl;
  f << "5, 6, coffee" << endl;
  f << "7, 8, confusion" << endl;
  f << "9, 10, hello" << endl;
  f << "11, 12, confusion" << endl;
  f << "13, 14, confusion" << endl;
  f.close();

  // Create tuples.
  data::DatasetInfo dd;
  arma::mat m;

  tuple<data::DatasetInfo, arma::mat> tuple1 = make_tuple(dd, m);
  tuple<decltype(tuple1), string> tuple2 = make_tuple(tuple1, filename);

  d.value = boost::any(tuple2);
  // Make sure it is not loaded yet.
  d.input = true;
  d.loaded = false;

  // Set up object to load into.
  tuple<data::DatasetInfo, arma::mat>* output = NULL;
  GetParam<tuple<data::DatasetInfo, arma::mat>>((const util::ParamData&) d,
      (void*) NULL, (void*) &output);

  BOOST_REQUIRE_EQUAL(get<0>(*output).Dimensionality(), 3);
  BOOST_REQUIRE_EQUAL(get<0>(*output).Type(0), data::Datatype::numeric);
  BOOST_REQUIRE_EQUAL(get<0>(*output).Type(1), data::Datatype::numeric);
  BOOST_REQUIRE_EQUAL(get<0>(*output).Type(2), data::Datatype::categorical);
  BOOST_REQUIRE_EQUAL(get<1>(*output).n_rows, 3);
  BOOST_REQUIRE_EQUAL(get<1>(*output).n_cols, 7);

  remove("test.csv");
}

BOOST_AUTO_TEST_CASE(GetParamModelTest)
{
  util::ParamData d;

  // Create value.
  string filename = "kernel.bin";
  kernel::GaussianKernel gk(5.0);
  data::Save("kernel.bin", "model", gk);

  // Create tuple.
  tuple<GaussianKernel, string> t = make_tuple(gk, filename);
  d.value = boost::any(t);
  // Make sure it is not loaded yet.
  d.input = true;
  d.loaded = false;

  tuple<GaussianKernel, string>* output = NULL;
  GetParam<tuple<GaussianKernel, string>>((const util::ParamData&) d,
      (void*) NULL, (void*) &output);

  BOOST_REQUIRE_EQUAL(get<0>(*output).Bandwidth(), 5.0);

  remove("kernel.bin");
}

BOOST_AUTO_TEST_SUITE_END();
