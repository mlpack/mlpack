/**
 * @file tests/imputation_test.cpp
 * @author Keon Kim
 *
 * Tests for Imputer class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace std;

/**
 * Make sure CustomImputation method replaces data 0 to 99.
 */
TEMPLATE_TEST_CASE("CustomImputationTest", "[ImputationTest]", float, double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  arma::Mat<ElemType> rowWiseInput(columnWiseInput);

  ElemType customValue = 99;
  ElemType mappedValue = 0.0;

  CustomImputation<ElemType> imputer(customValue);

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
 * Make sure CustomImputation method replaces NaNs with 99.
 */
TEMPLATE_TEST_CASE("CustomImputationNaNTest", "[ImputationTest]", float, double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  columnWiseInput.replace(0.0, std::numeric_limits<ElemType>::quiet_NaN());
  arma::Mat<ElemType> rowWiseInput(columnWiseInput);

  ElemType customValue = 99;
  ElemType mappedValue = std::numeric_limits<ElemType>::quiet_NaN();

  CustomImputation<ElemType> imputer(customValue);

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0 /*dimension*/, true);

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(99.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 3) == Approx(99.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(std::isnan(columnWiseInput(1, 2))); // We didn't impute in this row.
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
  REQUIRE(std::isnan(rowWiseInput(0, 3))); // We didn't impute in this column.
  REQUIRE(rowWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(std::isnan(rowWiseInput(1, 2))); // We didn't impute in this column.
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
TEMPLATE_TEST_CASE("MeanImputationTest", "[ImputationTest][tiny]", float,
    double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  arma::Mat<ElemType> rowWiseInput(columnWiseInput);

  ElemType mappedValue = 0.0;
  MeanImputation imputer;

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
 * Make sure MeanImputation method replaces NaNs with the mean value of each
 * dimension.
 */
TEMPLATE_TEST_CASE("MeanImputationNaNTest", "[ImputationTest]", float, double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  columnWiseInput.replace(0.0, std::numeric_limits<ElemType>::quiet_NaN());

  arma::Mat<ElemType> rowWiseInput(columnWiseInput);

  ElemType mappedValue = std::numeric_limits<ElemType>::quiet_NaN();

  MeanImputation imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 0, true);

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(2.5).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 3) == Approx(2.5).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(std::isnan(columnWiseInput(1, 2))); // We didn't impute in this row.
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
  REQUIRE(std::isnan(rowWiseInput(0, 3))); // We didn't impute in this column.
  REQUIRE(rowWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(std::isnan(rowWiseInput(1, 2))); // We didn't impute in this column.
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
TEMPLATE_TEST_CASE("MedianImputationTest", "[ImputationTest]", float, double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  arma::Mat<ElemType> rowWiseInput(columnWiseInput);
  ElemType mappedValue = 0.0;

  MedianImputation imputer;

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
 * Make sure MedianImputation method replaces NaNs with the median value of each
 * dimension.
 */
TEMPLATE_TEST_CASE("MedianImputationNaNTest", "[ImputationTest]", float, double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  columnWiseInput.replace(0.0, std::numeric_limits<ElemType>::quiet_NaN());
  arma::Mat<ElemType> rowWiseInput(columnWiseInput);
  ElemType mappedValue = std::numeric_limits<ElemType>::quiet_NaN();

  MedianImputation imputer;

  // column wise
  imputer.Impute(columnWiseInput, mappedValue, 1, true);

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(std::isnan(columnWiseInput(0, 1))); // We didn't impute in this row.
  REQUIRE(columnWiseInput(0, 2) == Approx(2.0).epsilon(1e-7));
  REQUIRE(std::isnan(columnWiseInput(0, 3))); // We didn't impute in this row.
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
  REQUIRE(std::isnan(rowWiseInput(0, 3))); // We didn't impute in this column.
  REQUIRE(rowWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(std::isnan(rowWiseInput(1, 2))); // We didn't impute in this column.
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
TEMPLATE_TEST_CASE("ListwiseDeletionTest", "[ImputationTest]", float, double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  arma::Mat<ElemType> rowWiseInput(columnWiseInput);
  ElemType mappedValue = 0.0;

  ListwiseDeletion imputer;

  imputer.Impute(columnWiseInput, mappedValue, 0, true); // column wise

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 1) == Approx(0.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 1) == Approx(4.0).epsilon(1e-7));

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
 * Make sure ListwiseDeletion method deletes the whole column (if column wise)
 * or the row (if row wise) if it contains NaNs.
 */
TEMPLATE_TEST_CASE("ListwiseDeletionNaNTest", "[ImputationTest]", float, double)
{
  typedef TestType ElemType;

  arma::Mat<ElemType> columnWiseInput("3.0 0.0 2.0 0.0;"
                                      "5.0 6.0 0.0 6.0;"
                                      "9.0 8.0 4.0 8.0;");
  columnWiseInput.replace(0.0, std::numeric_limits<ElemType>::quiet_NaN());
  arma::Mat<ElemType> rowWiseInput(columnWiseInput);
  ElemType mappedValue = std::numeric_limits<ElemType>::quiet_NaN();

  ListwiseDeletion imputer;

  imputer.Impute(columnWiseInput, mappedValue, 0, true); // column wise

  REQUIRE(columnWiseInput(0, 0) == Approx(3.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(0, 1) == Approx(2.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(1, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(std::isnan(columnWiseInput(1, 1))); // We didn't impute in this row.
  REQUIRE(columnWiseInput(2, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(columnWiseInput(2, 1) == Approx(4.0).epsilon(1e-7));

  imputer.Impute(rowWiseInput, mappedValue, 1, false); // row wise

  REQUIRE(rowWiseInput(0, 0) == Approx(5.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(0, 1) == Approx(6.0).epsilon(1e-7));
  REQUIRE(std::isnan(rowWiseInput(0, 2))); // We didn't impute in this column.
  REQUIRE(rowWiseInput(0, 3) == Approx(6.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 0) == Approx(9.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 1) == Approx(8.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 2) == Approx(4.0).epsilon(1e-7));
  REQUIRE(rowWiseInput(1, 3) == Approx(8.0).epsilon(1e-7));
}

/**
 * Test that Imputer throws an error when an invalid dimension is specified.
 */
TEST_CASE("ImputerInvalidDimensionTest", "[ImputationTest]")
{
  arma::mat x(10, 15, arma::fill::randu);

  Imputer<MeanImputation> imputer;

  REQUIRE_THROWS_AS(imputer.Impute(x, 0.0, 100), std::invalid_argument);
  REQUIRE_THROWS_AS(imputer.Impute(x, 0.0, 10), std::invalid_argument);

  REQUIRE_THROWS_AS(imputer.Impute(x, 0.0, 100, false), std::invalid_argument);
  REQUIRE_THROWS_AS(imputer.Impute(x, 0.0, 15, false), std::invalid_argument);
}
