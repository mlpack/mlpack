#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include <fstream>
#include <cstdio>

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(AnnTestFixture);

/**
 * Training Classification Model
 */
TEST_CASE_METHOD(AnnTestFixture, "AnnClassificationTest",
                 "[AnnMainTest][BindingTests]")
{
  arma::mat trainData(4, 100, arma::fill::randu); // 4 inputs, 100 samples
  
  // Use arma::mat for labels (doubles)
  arma::mat labels(1, 100);
  for (size_t i = 0; i < 100; i++) labels(0, i) = (double)(i % 2); // 2 classes

  arma::mat testData(4, 10, arma::fill::randu);

  // Create a simple architecture file
  std::ofstream arch("ann_test_arch_cls.txt");
  arch << "Linear 10" << std::endl;
  arch << "ReLU" << std::endl;
  arch << "Linear 2" << std::endl;
  arch << "LogSoftmax" << std::endl;
  arch.close();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("test", std::move(testData));
  SetInputParam("layers_file", std::string("ann_test_arch_cls.txt"));
  SetInputParam("max_iterations", (int) 10); // Fast test
  SetInputParam("input_dim", (int) 4);
  SetInputParam("output_dim", (int) 2);

  RUN_BINDING();

  // Check predictions presence and shape
  REQUIRE(params.Has("predictions"));

  arma::mat preds = params.Get<arma::mat>("predictions");
  REQUIRE(preds.n_cols == 10);
  REQUIRE(preds.n_rows == 1);
  // Cleanup
  remove("ann_test_arch_cls.txt");
}

/**
 * Checking Regression Model
 */
TEST_CASE_METHOD(AnnTestFixture, "AnnRegressionTest",
                 "[AnnMainTest][BindingTests]")
{
  arma::mat trainData(10, 100, arma::fill::randu);
  // Use arma::mat for labels (doubles)
  arma::mat labels(1, 100);
  for (size_t i = 0; i < 100; i++) labels(0, i) = (double)(i % 5);

  arma::mat testData(10, 10, arma::fill::randu);

  std::ofstream arch("ann_test_arch_reg.txt");
  arch << "Linear 10" << std::endl;
  arch << "ReLU" << std::endl;
  arch << "Linear 1" << std::endl;
  arch.close();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("test", std::move(testData));
  SetInputParam("layers_file", std::string("ann_test_arch_reg.txt"));
  SetInputParam("regression", true);
  SetInputParam("max_iterations", (int) 10);
  SetInputParam("input_dim", (int) 10);
  SetInputParam("output_dim", (int) 1);

  RUN_BINDING();

  REQUIRE(params.Has("predictions"));
  arma::mat preds = params.Get<arma::mat>("predictions");
  REQUIRE(preds.n_cols == 10);
  REQUIRE(preds.n_rows == 1);

  remove("ann_test_arch_reg.txt");
}

/**
 * Check automatic dimension inference (no -i or -O passed).
 */
TEST_CASE_METHOD(AnnTestFixture, "AnnAutoInferenceTest",
                 "[AnnMainTest][BindingTests]")
{
  arma::mat trainData(5, 50, arma::fill::randu);
  // Use arma::mat for labels
  arma::mat labels(1, 50);
  for (size_t i = 0; i < 50; i++) labels(0, i) = (double)(i % 2);

  std::ofstream arch("ann_test_arch_auto.txt");
  arch << "Linear 10" << std::endl;
  arch << "ReLU" << std::endl;
  arch << "Linear 2" << std::endl;
  arch << "LogSoftmax" << std::endl;
  arch.close();

  SetInputParam("training", std::move(trainData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("layers_file", std::string("ann_test_arch_auto.txt"));
  SetInputParam("max_iterations", (int) 5);

  RUN_BINDING();
  REQUIRE(params.Has("output_model")); // FFNModel

  remove("ann_test_arch_auto.txt");
}
