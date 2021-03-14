/**
 * @file tests/imputation_test.cpp
 * @author Keon Kim
 *
 * Tests for data::Imputer class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <sstream>

#include <mlpack/core.hpp>
#include <mlpack/core/data/load_impl.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>
#include <mlpack/core/data/map_policies/increment_policy.hpp>
#include <mlpack/core/data/map_policies/missing_policy.hpp>
#include <mlpack/core/data/imputer.hpp>
#include <mlpack/core/data/imputation_methods/custom_imputation.hpp>
#include <mlpack/core/data/imputation_methods/listwise_deletion.hpp>
#include <mlpack/core/data/imputation_methods/mean_imputation.hpp>
#include <mlpack/core/data/imputation_methods/median_imputation.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace std;

/**
 * 1. Make sure a CSV is loaded correctly with mappings using MissingPolicy.
 * 2. Try Imputer object with CustomImputation method to impute data "a".
 * (It is ok to test on one method since the other ones will be covered in the
 * next cases).
 */
TEST_CASE("DatasetMapperImputerTest", "[ImputationTest]")
{
  fstream f;
  f.open("test_file.csv", fstream::out);
  f << "a, 2, 3"  << endl;
  f << "5, 6, a"  << endl;
  f << "8, 9, 10" << endl;
  f.close();

  arma::mat input;
  MissingPolicy policy({"a"});
  DatasetMapper<MissingPolicy> info(policy);
  REQUIRE(data::Load("test_file.csv", input, info) == true);

  // row and column test.
  REQUIRE(input.n_rows == 3);
  REQUIRE(input.n_cols == 3);

  // Load check
  // MissingPolicy should convert strings to nans.
  REQUIRE(std::isnan(input(0, 0)) == true);
  REQUIRE(input(0, 1) == Approx(5.0).epsilon(1e-7));
  REQUIRE(input(0, 2) == Approx(8.0).epsilon(1e-7));
  REQUIRE(input(1, 0) == Approx(2.0).epsilon(1e-7));
  REQUIRE(input(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(input(1, 2) == Approx(9.0).epsilon(1e-7));
  REQUIRE(input(2, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(std::isnan(input(2, 1)) == true);
  REQUIRE(input(2, 2) == Approx(10.0).epsilon(1e-7));

  // convert missing vals to 99.
  CustomImputation<double> customStrategy(99);
  Imputer<double,
          DatasetMapper<MissingPolicy>,
          CustomImputation<double>> imputer(info, customStrategy);
  // convert a or nan to 99 for dimension 0.
  imputer.Impute(input, "a", 0);

  // Custom imputation result check.
  REQUIRE(input(0, 0) == Approx(99.0).epsilon(1e-7));
  REQUIRE(input(0, 1) == Approx(5.0).epsilon(1e-7));
  REQUIRE(input(0, 2) == Approx(8.0).epsilon(1e-7));
  REQUIRE(input(1, 0) == Approx(2.0).epsilon(1e-7));
  REQUIRE(input(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(input(1, 2) == Approx(9.0).epsilon(1e-7));
  REQUIRE(input(2, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(std::isnan(input(2, 1)) == true); // remains as NaN
  REQUIRE(input(2, 2) == Approx(10.0).epsilon(1e-7));

  // Remove the file.
  remove("test_file.csv");
}

/**
 * Make sure CustomImputation method replaces data 0 to 99.
 */
TEST_CASE("CustomImputationTest", "[ImputationTest]")
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double customValue = 99;
  double mappedValue = 0.0;

  CustomImputation<double> imputer(customValue);

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0/*dimension*/, true);

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(99.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 3) == Approx(99.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 2) == Approx(0.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 3) == Approx(8.0).epsilon(1e-7));

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false);

  REQUIRE(rowWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 1) == Approx(99.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 3) == Approx(0.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 2) == Approx(0.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 3) == Approx(8.0).epsilon(1e-7));
}

/**
 * Make sure MeanImputation method replaces data 0 to mean value of each
 * dimensions.
 */
TEST_CASE("MeanImputationTest", "[ImputationTest]")
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double mappedValue = 0.0;

  MeanImputation<double> imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0, true);

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(2.5).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 3) == Approx(2.5).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 2) == Approx(0.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 3) == Approx(8.0).epsilon(1e-7));

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false);

  REQUIRE(rowWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 1) == Approx(7.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 3) == Approx(0.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 2) == Approx(0.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 3) == Approx(8.0).epsilon(1e-7));
}

/**
 * Make sure MedianImputation method replaces data 0 to median value of each
 * dimensions.
 */
TEST_CASE("MedianImputationTest", "[ImputationTest]")
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double mappedValue = 0.0;

  MedianImputation<double> imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 1, true);

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(0.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 3) == Approx(0.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 2) == Approx(6.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 3) == Approx(8.0).epsilon(1e-7));

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false);

  REQUIRE(rowWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 1) == Approx(7.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 3) == Approx(0.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 2) == Approx(0.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(2, 3) == Approx(8.0).epsilon(1e-7));
}

/**
 * Make sure ListwiseDeletion method deletes the whole column (if column wise)
 * or the row (if row wise) containing value of 0.
 */
TEST_CASE("ListwiseDeletionTest", "[ImputationTest]")
{
  arma::mat columnWiseInput("3.0 0.0 2.0 0.0;"
                  "5.0 6.0 0.0 6.0;"
                  "9.0 8.0 4.0 8.0;");
  arma::mat rowWiseInput(columnWiseInput);
  double mappedValue = 0.0;

  ListwiseDeletion<double> imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0, true); // column wise

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 1) == Approx(0.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 1) == Approx(4.0).epsilon(1e-7));

  // row wise
  imputer.Impute(rowWiseInput, mappedValue, 1, false); // row wise

  REQUIRE(rowWiseInput(0, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 2) == Approx(0.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 3) == Approx(8.0).epsilon(1e-7));
}

/**
 * Make sure we can map non-strings.
 */
TEST_CASE("DatasetMapperNonStringMapping", "[ImputationTest]")
{
  IncrementPolicy incr(true);
  DatasetMapper<IncrementPolicy, double> dm(incr, 1);
  dm.MapString<size_t>(5.0, 0);
  dm.MapString<size_t>(4.3, 0);
  dm.MapString<size_t>(1.1, 0);

  REQUIRE(dm.NumMappings(0) == 3);

  REQUIRE(dm.Type(0) == data::Datatype::categorical);

  REQUIRE(dm.UnmapValue(5.0, 0) == 0);
  REQUIRE(dm.UnmapValue(4.3, 0) == 1);
  REQUIRE(dm.UnmapValue(1.1, 0) == 2);

  REQUIRE(dm.UnmapString(0, 0) == 5.0);
  REQUIRE(dm.UnmapString(1, 0) == 4.3);
  REQUIRE(dm.UnmapString(2, 0) == 1.1);
}

/**
 * Make sure we can map strange types.
 */
TEST_CASE("DatasetMapperPointerMapping", "[ImputationTest]")
{
  int a = 1, b = 2, c = 3;
  IncrementPolicy incr(true);
  DatasetMapper<IncrementPolicy, int*> dm(incr, 1);

  dm.MapString<size_t>(&a, 0);
  dm.MapString<size_t>(&b, 0);
  dm.MapString<size_t>(&c, 0);

  REQUIRE(dm.NumMappings(0) == 3);

  REQUIRE(dm.UnmapValue(&a, 0) == 0);
  REQUIRE(dm.UnmapValue(&b, 0) == 1);
  REQUIRE(dm.UnmapValue(&c, 0) == 2);

  REQUIRE(dm.UnmapString(0, 0) == &a);
  REQUIRE(dm.UnmapString(1, 0) == &b);
  REQUIRE(dm.UnmapString(2, 0) == &c);
}
