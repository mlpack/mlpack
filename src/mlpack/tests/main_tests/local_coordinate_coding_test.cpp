/**
 * @file tests/main_tests/local_coordinate_coding_test.cpp
 * @author Bhavya Bahl
 *
 * Test RUN_BINDING() of local_coordinate_coding_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LCCTestFixture);

/**
 * Ensure that the dimensions of encoded test points
 * and output dictionary are correct.
 */
TEST_CASE_METHOD(LCCTestFixture, "LCCDimensionsTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  int rows = x.n_rows, cols = x.n_cols;
  arma::mat t = x;
  int atoms = 10;

  SetInputParam("training", std::move(x));
  SetInputParam("test", std::move(t));
  SetInputParam("atoms", atoms);
  SetInputParam("max_iterations", (int) 2);

  RUN_BINDING();

  // Check that the output has correct dimensions.
  REQUIRE(params.Get<arma::mat>("codes").n_rows == (arma::uword) atoms);
  REQUIRE(params.Get<arma::mat>("codes").n_cols == (arma::uword) cols);
  REQUIRE(params.Get<arma::mat>("dictionary").n_rows == (arma::uword) rows);
  REQUIRE(params.Get<arma::mat>("dictionary").n_cols == (arma::uword) atoms);
}

/**
 * Ensure that trained model can be reused.
 */
TEST_CASE_METHOD(LCCTestFixture, "LCCOutputModelTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  arma::mat t = x;

  SetInputParam("training", std::move(x));
  SetInputParam("test", t);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);

  RUN_BINDING();

  // Get the encoded output and dictionary after training.
  arma::mat initCodes = std::move(params.Get<arma::mat>("codes"));
  arma::mat initDict = std::move(params.Get<arma::mat>("dictionary"));
  LocalCoordinateCoding* outputModel =
      std::move(params.Get<LocalCoordinateCoding*>("output_model"));

  ResetSettings();

  SetInputParam("input_model", std::move(outputModel));
  SetInputParam("test", std::move(t));

  RUN_BINDING();

  // Compare the output after reusing the trained model
  // to the original matrices.
  CheckMatrices(initCodes, params.Get<arma::mat>("codes"));
  CheckMatrices(initDict, params.Get<arma::mat>("dictionary"));
}

/**
 * Ensure that the number of rows in initial dictionary is same as 
 * the dimension of the points.
 */
TEST_CASE_METHOD(LCCTestFixture, "LCCInitDictTrainTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
  arma::mat initDict = {{1, 1}, {2, 2}, {3, 3}};

  SetInputParam("training", std::move(x));
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("atoms", (int) 2);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that the number of columns in initial dictionary is same as 
 * the number of atoms.
 */
TEST_CASE_METHOD(LCCTestFixture, "LCCInitDictAtomTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
  arma::mat initDict = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};

  SetInputParam("training", std::move(x));
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("atoms", (int) 3);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that training data and test data points
 * have same dimensionality.
 */
TEST_CASE_METHOD(LCCTestFixture, "LCCTrainAndTestDataDimTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  arma::mat t = x;

  t.shed_rows(1, 2);

  // Input data.
  SetInputParam("training", x);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);
  SetInputParam("test", std::move(t));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that only one out of training and input_model are specified.
 */
TEST_CASE_METHOD(LCCTestFixture, "LCCTrainAndInputModelTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");

  SetInputParam("training", x);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);

  RUN_BINDING();

  LocalCoordinateCoding* outputModel =
      std::move(params.Get<LocalCoordinateCoding*>("output_model"));

  // No need to input training data again.
  SetInputParam("input_model", std::move(outputModel));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that dimensionality of the trained model matches
 * the dimensionality of the test points.
 */
TEST_CASE_METHOD(LCCTestFixture, "LCCTrainedModelDimTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  arma:: mat t = x;
  t.shed_rows(1, 2);

  SetInputParam("training", x);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);

  RUN_BINDING();

  LocalCoordinateCoding* outputModel =
      std::move(params.Get<LocalCoordinateCoding*>("output_model"));

  SetInputParam("input_model", std::move(outputModel));
  SetInputParam("test", std::move(t));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
* Ensure that the number of atoms is positive and
* less than the number of training points.
*/
TEST_CASE_METHOD(LCCTestFixture, "LCCAtomsBoundTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
  SetInputParam("training", std::move(x));
  SetInputParam("atoms", (int) 5);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("atoms", (int) -1);
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
* Ensure that the program throws error for negative regularization parameter.
*/
TEST_CASE_METHOD(LCCTestFixture, "LCCNegativeLambdaTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
  SetInputParam("training", std::move(x));
  SetInputParam("atoms", (int) 2);
  SetInputParam("lambda", -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
* Ensure that the program throws error for negative tolerance.
*/
TEST_CASE_METHOD(LCCTestFixture, "LCCNegativeToleranceTest",
                 "[LCCMainTest][BindingTests]")
{
  arma::mat x = {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
  SetInputParam("training", std::move(x));
  SetInputParam("atoms", (int) 2);
  SetInputParam("tolerance", -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
* Ensure that the normalize parameter works.
*/
TEST_CASE_METHOD(LCCTestFixture, "LCCNormalizationTest",
                 "[LCCMainTest][BindingTests]")
{
  // Minimum required difference between the encodings of the test data.
  double delta = 1.0;

  arma::mat x = {{1, 2, 3, 4}, {2, 2, 3, 1}, {3, 2, 3, 0}, {1, 1, 4, 4}};
  arma::mat t = x;
  arma::mat initDict = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};

  SetInputParam("training", x);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initDict);
  SetInputParam("max_iterations", 2);
  SetInputParam("test", t);

  RUN_BINDING();

  arma::mat codes = std::move(params.Get<arma::mat>("codes"));

  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(x));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("max_iterations", (int) 2);
  SetInputParam("test", std::move(t));
  SetInputParam("normalize", (bool) 1);

  RUN_BINDING();

  double normDiff =
      arma::norm(params.Get<arma::mat>("codes") - codes, "fro");

  REQUIRE(normDiff > delta);
}

/*
* Ensure that changing max iterations changes the output.
*/
TEST_CASE_METHOD(LCCTestFixture, "LCCMaxIterTest",
                 "[LCCMainTest][BindingTests]")
{
  // Minimum required difference between the encodings of the test data.
  double delta = 1.0;

  arma::mat x = {{1, 2, 3, 4}, {2, 2, 3, 1}, {3, 2, 3, 0}, {1, 1, 4, 4}};
  arma::mat t = x;
  arma::mat initDict = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};


  SetInputParam("training", x);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initDict);
  SetInputParam("max_iterations", 2);
  SetInputParam("test", t);

  RUN_BINDING();
  arma::mat codes = std::move(params.Get<arma::mat>("codes"));

  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(x));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("max_iterations", (int) 4);
  SetInputParam("test", std::move(t));

  RUN_BINDING();

  double normDiff =
      arma::norm(params.Get<arma::mat>("codes") - codes, "fro");

  REQUIRE(normDiff > delta);
}

/*
* Ensure that changing tolerance changes the output.
*/
TEST_CASE_METHOD(LCCTestFixture, "LCCToleranceTest",
                 "[LCCMainTest][BindingTests]")
{
  // Minimum required difference between the encodings of the test data.
  double delta = 0.05;

  arma::mat x = {{1, 2, 3, 4}, {2, 2, 3, 1}, {3, 2, 3, 0}, {1, 1, 4, 4}};
  arma::mat t = x;
  arma::mat initDict = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};

  SetInputParam("training", x);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initDict);
  SetInputParam("test", t);
  SetInputParam("tolerance", (double) 0.01);

  RUN_BINDING();
  arma::mat codes = std::move(params.Get<arma::mat>("codes"));

  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(x));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("test", std::move(t));
  SetInputParam("tolerance", (double) 100.0);

  RUN_BINDING();

  double normDiff =
      arma::norm(params.Get<arma::mat>("codes") - codes, "fro");

  REQUIRE(normDiff > delta);
}

/*
* Ensure that changing regularization parameter changes the output.
*/
TEST_CASE_METHOD(LCCTestFixture, "LCCLambdaTest",
                 "[LCCMainTest][BindingTests]")
{
  // Minimum required difference between the encodings of the test data.
  double delta = 1.0;

  arma::mat x = {{1, 2, 3, 4}, {2, 2, 3, 1}, {3, 2, 3, 0}, {1, 1, 4, 4}};
  arma::mat t = x;
  arma::mat initDict = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};

  SetInputParam("training", x);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initDict);
  SetInputParam("test", t);
  SetInputParam("lambda", (double) 0.0);

  RUN_BINDING();
  arma::mat codes = std::move(params.Get<arma::mat>("codes"));

  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(x));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("test", std::move(t));
  SetInputParam("lambda", (double) 1.0);

  RUN_BINDING();

  double normDiff =
      arma::norm(params.Get<arma::mat>("codes") - codes, "fro");

  REQUIRE(normDiff > delta);
}
