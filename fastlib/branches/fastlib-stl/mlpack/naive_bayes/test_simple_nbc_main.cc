#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>
#include "simple_nbc.h"



#define BOOST_TEST_MODULE Test_Simple_NBC_Main
#include <boost/test/unit_test.hpp>

PROGRAM_INFO("NBC", "Tests the simple nbc class.", "nbc");
using namespace mlpack;

BOOST_AUTO_TEST_CASE(MainTest) { 
    const char *train_datab = "trainSet.arff";
    const char *train_resb = "trainRes.arff";
    const char *test_datab = "testSet.arff";
    const char *test_resb = "testRes.arff";
    const int num_classesb = 2;

    const char *filename_train_, *filename_test_;
    const char *train_result_, *test_result_;
    size_t number_of_classes_;

    filename_train_ = train_datab;
    filename_test_ = test_datab;
    train_result_ = train_resb;
    test_result_ = test_resb;
    number_of_classes_ = num_classesb;

    arma::mat train_data, train_res, calc_mat;
    data::Load(filename_train_, train_data);
    data::Load(train_result_, train_res);

    IO::GetParam<int>("nbc/classes") = number_of_classes_;
    SimpleNaiveBayesClassifier nbc_test_(train_data);

    size_t number_of_features = nbc_test_.means_.n_rows;
    calc_mat.zeros(2*number_of_features + 1, number_of_classes_);

    for(size_t i = 0; i < number_of_features; i++) {
     for(size_t j = 0; j < number_of_classes_; j++) {
        calc_mat(i, j) = nbc_test_.means_(i, j);
        calc_mat(i + number_of_features, j) = nbc_test_.variances_(i, j);
      }
    }
    for(size_t i = 0; i < number_of_classes_; i++) {
      calc_mat(2 * number_of_features, i) = nbc_test_.class_probabilities_(i);
    }

    for(size_t i = 0; i < calc_mat.n_rows; i++) {
      for(size_t j = 0; j < number_of_classes_; j++) {
        BOOST_REQUIRE_CLOSE(train_res(i, j) + .00001, calc_mat(i, j), .01);
       }
    }

    arma::mat test_data, test_res;
    arma::vec test_res_vec, calc_vec;
    data::Load(filename_test_, test_data);
    data::Load(test_result_, test_res);

    nbc_test_.Classify(test_data, calc_vec);

    size_t number_of_datum = test_data.n_cols;
    test_res_vec = test_res.col(0);

    for(size_t i = 0; i < number_of_datum; i++) {
      assert(test_res_vec(i) == calc_vec(i));
    }

}
