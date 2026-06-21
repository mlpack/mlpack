/**
 * @file methods/softmax_regression/softmax_regression_utils_impl.hpp
 * @author Dirk Eddelbuettel
 *
 * Implementation of prediction step used in 'classify' and 'probabilities'.
 * Used only for bindings.
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

template<typename Model>
inline void RunPredictionStep(util::Params& params,
                              util::Timers& timers,
                              const size_t numClasses,
                              const Model& model,
                              const bool retPreds,  // no default argument here
                              const bool retProbas) // to have diff. signature
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
  if (retPreds)
    params.Get<arma::Row<size_t>>("predictions") = std::move(predictLabels);
  // Save probabiltities if requested
  if (retProbas)
    params.Get<arma::mat>("probabilities") = std::move(probabilities);
}

} // namespace mlpack

#endif
