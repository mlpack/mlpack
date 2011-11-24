/**
 * @file simple_nbc.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  It is assumed that the features have been sampled from a
 * Gaussian PDF.
 */
#include <mlpack/core.h>

#include "simple_nbc.hpp"

namespace mlpack {
namespace naive_bayes {

SimpleNaiveBayesClassifier::SimpleNaiveBayesClassifier(const arma::mat& data)
{
  size_t number_examples = data.n_cols;
  size_t number_features = data.n_rows - 1;

  arma::vec feature_sum, feature_sum_squared;
  feature_sum.zeros(number_features);
  feature_sum_squared.zeros(number_features);

  // Update the variables, private and local, according to the number of
  // features and classes present in the data.
  number_of_classes_ = mlpack::CLI::GetParam<int>("nbc/classes");
  class_probabilities_.set_size(number_of_classes_);
  means_.set_size(number_features,number_of_classes_);
  variances_.set_size(number_features,number_of_classes_);

  Log::Info << number_examples << " examples with " << number_features
      << " features each" << std::endl;

  CLI::GetParam<int>("nbc/features") = number_features;
  CLI::GetParam<int>("nbc/examples") = number_examples;

  // Calculate the class probabilities as well as the sample mean and variance
  // for each of the features with respect to each of the labels.
  for (size_t i = 0; i < number_of_classes_; i++ )
  {
    size_t number_of_occurrences = 0;
    for (size_t j = 0; j < number_examples; j++)
    {
      size_t flag = (size_t)  data(number_features, j);
      if (i == flag)
      {
        ++number_of_occurrences;
        for (size_t k = 0; k < number_features; k++)
        {
          double tmp = data(k, j);
          feature_sum(k) += tmp;
          feature_sum_squared(k) += tmp*tmp;
        }
      }
    }

    class_probabilities_[i] = (double) number_of_occurrences
        / (double) number_examples;

    for (size_t k = 0; k < number_features; k++)
    {
      double sum = feature_sum(k);
      double sum_squared = feature_sum_squared(k);

      means_(k, i) = (sum / number_of_occurrences);
      variances_(k, i) = (sum_squared - (sum * sum / number_of_occurrences))
          / (number_of_occurrences - 1);
    }

    // Reset the summations to zero for the next iteration
    feature_sum.zeros(number_features);
    feature_sum_squared.zeros(number_features);
  }
}

void SimpleNaiveBayesClassifier::Classify(const arma::mat& test_data,
                                          arma::vec& results)
{
  // Check that the number of features in the test data is same as in the
  // training data.
  Log::Assert(test_data.n_rows - 1 == means_.n_rows);

  arma::vec tmp_vals(number_of_classes_);
  size_t number_features = test_data.n_rows - 1;

  results.zeros(test_data.n_cols);

  Log::Info << test_data.n_cols << " test cases with " << number_features
      << " features each" << std::endl;

  CLI::GetParam<int>("nbc/tests") = test_data.n_cols;
  // Calculate the joint probability for each of the data points for each of the
  // classes.

  // Loop over every test case.
  for (size_t n = 0; n < test_data.n_cols; n++)
  {
    // Loop over every class.
    for (size_t i = 0; i < number_of_classes_; i++)
    {
      // Use the log values to prevent floating point underflow.
      tmp_vals(i) = log(class_probabilities_(i));

      // Loop over every feature.
      for (size_t j = 0; j < number_features; j++)
      {
        tmp_vals(i) += log(gmm::phi(test_data(j, n), means_(j, i),
            variances_(j, i)));
      }
    }

    // Find the index of the maximum value in tmp_vals.
    size_t max = 0;
    for (size_t k = 0; k < number_of_classes_; k++)
    {
      if (tmp_vals(max) < tmp_vals(k))
        max = k;
    }
    results(n) = max;
  }

  return;
}

}; // namespace naive_bayes
}; // namespace mlpack
