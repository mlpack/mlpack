/*
 * @file adaboost_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of the AdaBoost class
 *
 */

#ifndef _MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP
#define _MLPACK_METHODS_ADABOOST_ADABOOST_IMPL_HPP

#include "adaboost.hpp"

namespace mlpack {
namespace adaboost {

template<typename MatType, typename WeakLearner>
Adaboost<MatType, WeakLearner>::Adaboost(const MatType& data, const arma::Row<size_t>& labels,
         int iterations, size_t classes, const WeakLearner& other)
{
  int j, i;
  
  // load the initial weights
  
  const double initWeight = 1 / (data.n_cols * classes);
  arma::Row<double> D(data.n_cols);
  D.fill(initWeight);

  double rt, alphat = 0.0, zt;
  arma::Row<size_t> predictedLabels(labels.n_cols);
  MatType tempData(data);
  // now start the boosting rounds
  for (i = 0; i < iterations; i++)
  {
    rt = 0.0;
    zt = 0.0;

    //transform data, as per rules for perceptron
    for (j = 0;j < tempData.n_cols;j++)
      tempData.col(i) = D(i) * tempData.col(i);

    // for now, perceptron initialized with default parameters
    //mlpack::perceptron::Perceptron<> p(tempData, labels, 1000);
    WeakLearner w(other);
    w.Classify(tempData, predictedLabels);

    // Now, start calculation of alpha(t)

    // building a helper rowvector, mispredict to help in calculations.
    // this stores the value of Yi(l)*ht(xi,l)
    
    arma::Row<double> mispredict(predictedLabels.n_cols);
    
    for(j = 0;j < predictedLabels.n_cols; j++)
    {
      if (predictedLabels(j) != labels(j))
        mispredict(j) = -predictedLabels(j);
      else
        mispredict(j) = predictedLabels(j);
    }

    // begin calculation of rt

    for (j = 0;j < predictedLabels.n_cols; j++)
      rt +=(D(j) * mispredict(j));

    // end calculation of rt

    alphat = 0.5 * log((1 + rt) / (1 - rt));

    // end calculation of alphat
    
    for (j = 0;j < mispredict.n_cols; j++)
    {
      zt += D(i) * exp(-1 * alphat * mispredict(i));
      D(i) = D(i) * exp(-1 * alphat * mispredict(i));
    }

    D = D / zt;

  }

}

} // namespace adaboost
} // namespace mlpack
#endif