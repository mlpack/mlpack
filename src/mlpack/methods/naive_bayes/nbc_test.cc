#include <mlpack/core.h>
#include "simple_nbc.h"

#define BOOST_TEST_MODULE Test_Simple_NBC_Main
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace naive_bayes;

BOOST_AUTO_TEST_CASE(SimpleNBCTest) {
  const char* filename_train_ = "trainSet.csv";
  const char* filename_test_ = "testSet.csv";
  const char* train_result_ = "trainRes.csv";
  const char* test_result_ = "testRes.csv";
  size_t number_of_classes_ = 2;

  arma::mat train_data, train_res, calc_mat;
  data::Load(filename_train_, train_data, true);
  data::Load(train_result_, train_res, true);

  CLI::GetParam<int>("nbc/classes") = number_of_classes_;
  SimpleNaiveBayesClassifier nbc_test_(train_data);

  size_t number_of_features = nbc_test_.means_.n_rows;
  calc_mat.zeros(2 * number_of_features + 1, number_of_classes_);

  for (size_t i = 0; i < number_of_features; i++) {
    for (size_t j = 0; j < number_of_classes_; j++) {
      calc_mat(i, j) = nbc_test_.means_(i, j);
      calc_mat(i + number_of_features, j) = nbc_test_.variances_(i, j);
    }
  }
  for (size_t i = 0; i < number_of_classes_; i++)
    calc_mat(2 * number_of_features, i) = nbc_test_.class_probabilities_(i);

  for(size_t i = 0; i < calc_mat.n_rows; i++) {
    for(size_t j = 0; j < number_of_classes_; j++) {
      BOOST_REQUIRE_CLOSE(train_res(i, j) + .00001, calc_mat(i, j), .01);
    }
  }

  arma::mat test_data, test_res;
  arma::vec test_res_vec, calc_vec;
  data::Load(filename_test_, test_data, true);
  data::Load(test_result_, test_res, true);

  nbc_test_.Classify(test_data, calc_vec);

  size_t number_of_datum = test_data.n_cols;
  test_res_vec = test_res.col(0);

  for(size_t i = 0; i < number_of_datum; i++)
    BOOST_REQUIRE_EQUAL(test_res_vec(i), calc_vec(i));
}
