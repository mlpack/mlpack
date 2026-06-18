/**
 * @file methods/softmax_regression/softmax_regression_utils_impl.hpp
 * @author Dirk Eddelbuettel
 *
 * Implementation of function to be optimized for softmax regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_UTILS_IMPL_HPP
#define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_UTILS_IMPL_HPP

#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

namespace mlpack {
namespace smutil {

inline size_t CalculateNumberOfClasses(const size_t numClasses,
                                       const arma::Row<size_t>& trainLabels)
{
  if (numClasses == 0)
  {
    const set<size_t> unique_labels(begin(trainLabels),
                                    end(trainLabels));
    return unique_labels.size();
  }
  else
  {
    return numClasses;
  }
}

template<typename Model>
inline void TestClassifyAcc(util::Params& params,
                            util::Timers& timers,
                            const size_t numClasses,
                            const Model& model)
{
  using namespace mlpack;

  // Get the test dataset, and get predictions.
  arma::mat testData = std::move(params.Get<arma::mat>("test"));

  arma::Row<size_t> predictLabels;
  arma::mat probabilities;
  timers.Start("softmax_regression_classification");
  model.Classify(testData, predictLabels, probabilities);
  timers.Stop("softmax_regression_classification");

  // Calculate accuracy, if desired.
  if (params.Has("test_labels"))
  {
    arma::Row<size_t> testLabels =
      std::move(params.Get<arma::Row<size_t>>("test_labels"));

    if (testData.n_cols != testLabels.n_elem)
    {
      Log::Fatal << "Test data given with " << PRINT_PARAM_STRING("test")
          << " has " << testData.n_cols << " points, but labels in "
          << PRINT_PARAM_STRING("test_labels") << " have " << testLabels.n_elem
          << " labels!" << endl;
    }

    vector<size_t> bingoLabels(numClasses, 0);
    vector<size_t> labelSize(numClasses, 0);
    for (arma::uword i = 0; i != predictLabels.n_elem; ++i)
    {
      if (predictLabels(i) == testLabels(i))
      {
        ++bingoLabels[testLabels(i)];
      }
      ++labelSize[testLabels(i)];
    }

    size_t totalBingo = 0;
    for (size_t i = 0; i != bingoLabels.size(); ++i)
    {
      Log::Info << "Accuracy for points with label " << i << " is "
          << (bingoLabels[i] / static_cast<double>(labelSize[i])) << " ("
          << bingoLabels[i] << " of " << labelSize[i] << ")." << endl;
      totalBingo += bingoLabels[i];
    }

    Log::Info << "Total accuracy for all points is "
        << (totalBingo) / static_cast<double>(predictLabels.n_elem) << " ("
        << totalBingo << " of " << predictLabels.n_elem << ")." << endl;
  }
  // Save predictions if requested
  if (params.Has("predictions"))
    params.Get<arma::Row<size_t>>("predictions") = std::move(predictLabels);
  // Save probabiltities if requested
  if (params.Has("probabilities"))
    params.Get<arma::mat>("probabilities") = std::move(probabilities);
}

template<typename Model>
inline Model* TrainSoftmax(util::Params& params,
                           util::Timers& timers,
                           const size_t maxIterations)
{
  using namespace mlpack;

  arma::mat trainData = std::move(params.Get<arma::mat>("training"));
  arma::Row<size_t> trainLabels =
    std::move(params.Get<arma::Row<size_t>>("labels"));

  if (trainData.n_cols != trainLabels.n_elem)
    Log::Fatal << "Samples of input_data should same as the size of "
        << "input_label." << endl;

  const size_t numClasses = smutil::CalculateNumberOfClasses(
      (size_t) params.Get<int>("number_of_classes"), trainLabels);

  const bool intercept = params.Has("no_intercept") ? false : true;

  const size_t numBasis = 5;
  ens::L_BFGS optimizer(numBasis, maxIterations);
  timers.Start("softmax_regression_optimization");
  Model* sm = new Model(trainData, trainLabels, numClasses,
      params.Get<double>("lambda"), intercept, std::move(optimizer));
  timers.Stop("softmax_regression_optimization");
  return sm;
}

} // namespace smutil
} // namespace mlpack

#endif
