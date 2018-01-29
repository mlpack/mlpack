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
  CLI::ClearSettings();
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
  d.noTranspose = false;

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
  d.noTranspose = false;

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
  d.noTranspose = false;

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
  d.noTranspose = false;

  // Set up object to load into.
  tuple<data::DatasetInfo, arma::mat>* output = NULL;
  GetParam<tuple<data::DatasetInfo, arma::mat>>((const util::ParamData&) d,
      (void*) NULL, (void*) &output);

  BOOST_REQUIRE_EQUAL(get<0>(*output).Dimensionality(), 3);
  BOOST_REQUIRE_EQUAL((int) get<0>(*output).Type(0),
      (int) data::Datatype::numeric);
  BOOST_REQUIRE_EQUAL((int) get<0>(*output).Type(1),
      (int) data::Datatype::numeric);
  BOOST_REQUIRE_EQUAL((int) get<0>(*output).Type(2),
      (int) data::Datatype::categorical);
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
  tuple<GaussianKernel*, string> t = make_tuple((GaussianKernel*) NULL,
      filename);
  d.value = boost::any(t);
  // Make sure it is not loaded yet.
  d.input = true;
  d.loaded = false;

  GaussianKernel** output = NULL;
  GetParam<GaussianKernel*>((const util::ParamData&) d, (void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL((*output)->Bandwidth(), 5.0);

  remove("kernel.bin");
  delete *output;
}

BOOST_AUTO_TEST_CASE(RawParamDoubleTest)
{
  // This should function the same as GetParam for doubles.
  util::ParamData d;
  double x = 5.0;
  d.value = boost::any(x);

  double* output = NULL;
  GetParam<double>((const util::ParamData&) d, (const void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL(*output, 5.0);
}

BOOST_AUTO_TEST_CASE(RawParamMatTest)
{
  // This should return the matrix as-is without loading.
  util::ParamData d;
  // Create value.
  string filename = "hello.csv";
  arma::mat m(5, 5, arma::fill::ones);
  tuple<arma::mat, string> tuple = make_tuple(m, filename);
  d.value = boost::any(tuple);
  d.input = true;
  d.loaded = false;
  d.noTranspose = false;

  arma::mat* output = NULL;
  GetRawParam<arma::mat>((const util::ParamData&) d, (void*) NULL,
      (void*) &output);

  BOOST_REQUIRE_EQUAL(output->n_rows, 5);
  BOOST_REQUIRE_EQUAL(output->n_cols, 5);
  for (size_t i = 0; i < 25; ++i)
    BOOST_REQUIRE_EQUAL((*output)[i], 1.0);
}

BOOST_AUTO_TEST_CASE(GetRawParamModelTest)
{
  util::ParamData d;

  // Create value.
  string filename = "kernel.bin";
  kernel::GaussianKernel gk(5.0);

  // Create tuple.
  tuple<GaussianKernel*, string> t = make_tuple(&gk, filename);
  d.value = boost::any(t);
  // Make sure it is not loaded yet.
  d.input = true;
  d.loaded = false;

  tuple<GaussianKernel*, string>* output = NULL;
  GetRawParam<tuple<GaussianKernel*, string>>((const util::ParamData&) d,
      (void*) NULL, (void*) &output);

  BOOST_REQUIRE_EQUAL(get<0>(*output)->Bandwidth(), 5.0);
}

BOOST_AUTO_TEST_CASE(GetRawParamDatasetInfoTest)
{
  util::ParamData d;

  // Create value.
  string filename = "test.csv";

  // Create tuples.
  data::DatasetInfo dd(3);
  arma::mat m(3, 3, arma::fill::randu);

  tuple<data::DatasetInfo, arma::mat> tuple1 = make_tuple(dd, m);
  tuple<decltype(tuple1), string> tuple2 = make_tuple(tuple1, filename);

  d.value = boost::any(tuple2);
  // Make sure it is not loaded yet.
  d.input = true;
  d.loaded = false;
  d.noTranspose = false;

  // Set up object to load into.
  tuple<data::DatasetInfo, arma::mat>* output = NULL;
  GetRawParam<tuple<data::DatasetInfo, arma::mat>>((const util::ParamData&) d,
      (void*) NULL, (void*) &output);

  BOOST_REQUIRE_EQUAL(get<0>(*output).Dimensionality(), 3);
  BOOST_REQUIRE_EQUAL(get<1>(*output).n_rows, 3);
  BOOST_REQUIRE_EQUAL(get<1>(*output).n_cols, 3);
}

// Check that we can successfully write a matrix to file.
BOOST_AUTO_TEST_CASE(OutputParamMatTest)
{
  util::ParamData d;

  // Create value.
  string filename = "test.csv";
  arma::mat m(3, 3, arma::fill::randu);
  tuple<arma::mat, string> t = make_tuple(m, filename);

  d.value = boost::any(t);
  d.input = false;
  d.noTranspose = false;

  // Now save it.
  OutputParam<arma::mat>((const util::ParamData&) d, (const void*) NULL,
      (void*) NULL);

  arma::mat m2;
  BOOST_REQUIRE(data::Load("test.csv", m2));

  CheckMatrices(m, m2);

  remove("test.csv");
}

// Check that we can successfully write an unsigned matrix to file.
BOOST_AUTO_TEST_CASE(OutputParamUmatTest)
{
  util::ParamData d;

  // Create value.
  string filename = "test.csv";
  arma::Mat<size_t> m(3, 3, arma::fill::randu);
  tuple<arma::Mat<size_t>, string> t = make_tuple(m, filename);

  d.value = boost::any(t);
  d.input = false;
  d.noTranspose = false;

  // Now save it.
  OutputParam<arma::Mat<size_t>>((const util::ParamData&) d, (const void*) NULL,
      (void*) NULL);

  arma::Mat<size_t> m2;
  BOOST_REQUIRE(data::Load("test.csv", m2));

  CheckMatrices(m, m2);

  remove("test.csv");
}

// Check that we can successfully write a model to file.
BOOST_AUTO_TEST_CASE(OutputParamModelTest)
{
  util::ParamData d;

  // Create value.
  string filename = "kernel.bin";
  GaussianKernel gk(5.0);
  tuple<GaussianKernel*, string> t = make_tuple(&gk, filename);

  d.value = boost::any(t);
  d.input = false;

  // Now save it.
  OutputParam<GaussianKernel>((const util::ParamData&) d, (const void*) NULL,
      (void*) NULL);

  GaussianKernel gk2(1.0);
  BOOST_REQUIRE(data::Load("kernel.bin", "model", gk2));

  BOOST_REQUIRE_EQUAL(gk.Bandwidth(), gk2.Bandwidth());

  remove("kernel.bin");
}

// Test setting a primitive type parameter.
BOOST_AUTO_TEST_CASE(SetParamDoubleTest)
{
  util::ParamData d;

  // Create initial value.
  double dd = 5.0;
  d.value = boost::any(dd);

  // Now create second value.
  double dd2 = 1.0;
  boost::any a(dd2);
  SetParam<double>((const util::ParamData&) d, (const void*) &a, (void*) NULL);

  // Make sure it's the right thing.
  double* dd3 = NULL;
  GetParam<double>((const util::ParamData&) d, (const void*) NULL,
      (void*) &dd3);

  BOOST_REQUIRE_EQUAL((*dd3), dd2);
}

// Test that setting a flag works.
BOOST_AUTO_TEST_CASE(SetParamBoolTest)
{
  util::ParamData d;

  // Create initial value.
  bool b = false;
  d.value = boost::any(b);
  d.wasPassed = true;

  // Now create second value.
  bool b2 = true;
  boost::any a(b2);
  SetParam<bool>((const util::ParamData&) d, (const void*) &a, (void*) NULL);

  BOOST_REQUIRE_EQUAL(boost::any_cast<bool>(d.value), true);
}

// Test that calling SetParam on a matrix sets the string correctly.
BOOST_AUTO_TEST_CASE(SetParamMatrixTest)
{
  util::ParamData d;

  // Create initial value.
  string filename = "hello.csv";
  arma::mat m(5, 5, arma::fill::randu);
  d.value = boost::any(make_tuple(m, filename));

  // Get a new string.
  string newFilename = "new.csv";
  boost::any a2(newFilename);

  SetParam<arma::mat>((const util::ParamData&) d, (const void*) &a2,
      (void*) NULL);

  // Make sure the change went through.
  tuple<arma::mat, string>& t =
      *boost::any_cast<tuple<arma::mat, string>>(&d.value);
  BOOST_REQUIRE_EQUAL(get<1>(t), "new.csv");
}

// Test that calling SetParam on a model sets the string correctly.
BOOST_AUTO_TEST_CASE(SetParamModelTest)
{
  util::ParamData d;

  // Create initial value.
  string filename = "kernel.bin";
  GaussianKernel gk(2.0);
  d.value = boost::any(make_tuple(&gk, filename));

  // Get a new string.
  string newFilename = "new_kernel.bin";
  boost::any a2(newFilename);

  SetParam<GaussianKernel>((const util::ParamData&) d, (const void*) &a2,
      (void*) NULL);

  // Make sure the change went through.
  tuple<GaussianKernel*, string>& t =
      *boost::any_cast<tuple<GaussianKernel*, string>>(&d.value);

  BOOST_REQUIRE_EQUAL(get<1>(t), "new_kernel.bin");
}

// Test that calling SetParam on a mat/DatasetInfo successfully sets the
// filename.
BOOST_AUTO_TEST_CASE(SetParamDatasetInfoMatTest)
{
  util::ParamData d;

  // Create initial value.
  using namespace data;
  string filename = "test.csv";
  arma::mat m(3, 3, arma::fill::randu);
  DatasetInfo di(3);
  tuple<DatasetInfo, arma::mat> t1 = make_tuple(di, m);
  tuple<tuple<DatasetInfo, arma::mat>, string> t2 = make_tuple(t1, filename);
  d.value = boost::any(t2);
  d.noTranspose = false;

  // Now get new filename.
  string newFilename = "new_filename.csv";
  boost::any a2(newFilename);

  SetParam<tuple<DatasetInfo, arma::mat>>((const util::ParamData&) d,
      (const void*) &a2, (void*) NULL);

  // Check that the name is right.
  tuple<tuple<DatasetInfo, arma::mat>, string>& t3 =
      *boost::any_cast<tuple<tuple<DatasetInfo, arma::mat>, string>>(&d.value);

  BOOST_REQUIRE_EQUAL(get<1>(t3), "new_filename.csv");
}

// Test that GetAllocatedMemory() will properly return NULL for a non-model
// type.
BOOST_AUTO_TEST_CASE(GetAllocatedMemoryNonModelTest)
{
  util::ParamData d;

  bool b = true;
  d.value = boost::any(b);
  d.input = true;

  void* result = (void*) 1; // Invalid pointer, should be overwritten.

  GetAllocatedMemory<bool>((const util::ParamData&) d,
      (const void*) NULL, (void*) &result);

  BOOST_REQUIRE_EQUAL(result, (void*) NULL);

  // Also test with a matrix type.
  arma::mat test(10, 10, arma::fill::ones);
  string filename = "test.csv";
  tuple<arma::mat, string> t = make_tuple(test, filename);
  d.value = boost::any(t);

  result = (void*) 1;

  GetAllocatedMemory<arma::mat>((const util::ParamData&) d,
      (const void*) NULL, (void*) &result);

  BOOST_REQUIRE_EQUAL(result, (void*) NULL);
}

// Test that GetAllocatedMemory() will properly return pointers for a
// serializable model type.
BOOST_AUTO_TEST_CASE(GetAllocatedMemoryModelTest)
{
  util::ParamData d;

  GaussianKernel g(2.0);
  string filename = "hello.bin";
  tuple<GaussianKernel*, string> t = make_tuple(&g, filename);
  d.value = boost::any(t);
  d.input = true;

  void* result = NULL;

  GetAllocatedMemory<GaussianKernel*>((const util::ParamData&) d,
      (const void*) NULL, (void*) &result);

  BOOST_REQUIRE_EQUAL(&g, (GaussianKernel*) result);
}

// Test that calling DeleteAllocatedMemory() on non-model types does not delete
// pointers.
BOOST_AUTO_TEST_CASE(DeleteAllocatedMemoryNonModelTest)
{
  util::ParamData d;

  bool b = true;
  d.value = boost::any(b);
  d.input = true;

  DeleteAllocatedMemory<bool>((const util::ParamData&) d,
      (const void*) NULL, (void*) NULL);

  arma::mat test(10, 10, arma::fill::ones);
  string filename = "test.csv";
  tuple<arma::mat, string> t = make_tuple(test, filename);
  d.value = boost::any(t);

  DeleteAllocatedMemory<arma::mat>((const util::ParamData&) d,
      (const void*) NULL, (void*) NULL);
}

// Test that DeleteAllocatedMemory() will properly delete pointers for a
// serializable model type.
BOOST_AUTO_TEST_CASE(DeleteAllocatedMemoryModelTest)
{
  // This test will just delete it, and we'll hope that it worked and that
  // valgrind won't throw any issues (so really we can't *quite* test this in
  // the context of the boost unit test framework).
  util::ParamData d;

  GaussianKernel* g = new GaussianKernel(2.0);
  string filename = "hello.bin";
  tuple<GaussianKernel*, string> t = make_tuple(g, filename);

  d.value = boost::any(t);
  d.input = false;

  DeleteAllocatedMemory<GaussianKernel*>((const util::ParamData&) d,
      (const void*) NULL, (void*) NULL);
}

BOOST_AUTO_TEST_SUITE_END();
