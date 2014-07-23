/**
 * @file Adaboost_test.cpp
 * @author Udit Saxena
 *
 * Tests for Adaboost class.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/adaboost/adaboost.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::adaboost;

BOOST_AUTO_TEST_SUITE(AdaboostTest);

BOOST_AUTO_TEST_CASE(IrisSet)
{
  arma::mat inputData;

  if (!data::Load("iris.txt", inputData))
    BOOST_FAIL("Cannot load test dataset iris.txt!");

  arma::Mat<size_t> labels;

  if (!data::Load("iris_labels.txt",labels))
    BOOST_FAIL("Cannot load labels for iris iris_labels.txt");
  
  // no need to map the labels here

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptron_iter iterations.
  int perceptron_iter = 4000;

  perceptron::Perceptron<> p(inputData, labels.row(0), perceptron_iter);

  // Define parameters for the adaboost
  int iterations = 15;
  int classes = 3;
  Adaboost<> a(inputData, labels.row(0), iterations, classes, p);
  int countError = 0;
  for (size_t i = 0; i < labels.n_cols; i++)
    if(labels(i) != a.finalHypothesis(i))
    { 
      std::cout<<i<<" prediction not correct!\n";
      countError++;
    }
  std::cout<<"\nFinally - There are "<<countError<<" number of misclassified records.\n";  
  std::cout<<"The error rate is: "<<(double)countError * 100/labels.n_cols<<"%\n";
}
BOOST_AUTO_TEST_SUITE_END();