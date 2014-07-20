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

//
// Currently this is an implementation of adaboost.m1
// which will be templatized later and adaboost.mh and
// adaboost.samme will be added.
// 

template<typename MatType, typename WeakLearner>
Adaboost<MatType, WeakLearner>::Adaboost(const MatType& data, const arma::Row<size_t>& labels,
         int iterations, size_t classes, const WeakLearner& other)
{
  int j, i;
  
  // note: put a fail safe for classes or remove it entirely
  // by using unique function.

  // load the initial weights
  
  const double initWeight = 1 / (data.n_cols * classes);
  arma::rowvec D(data.n_cols);
  D.fill(initWeight);

  size_t countMP; // for counting mispredictions.
  double rt, alphat = 0.0, zt, et;
  arma::Row<size_t> predictedLabels(labels.n_cols);
  MatType tempData(data);
  
  // This behaves as ht(x)
  arma::rowvec mispredict(predictedLabels.n_cols);

  arma::mat sumFinalH(data.n_cols, classes);
  sumFinalH.fill(0.0);

  arma::rowvec finalH(labels.n_cols);
  // now start the boosting rounds
  for (i = 0; i < iterations; i++)
  {
    countMP = 0;
    rt = 0.0;
    zt = 0.0;
    
    // call the other weak learner and train the labels.
    WeakLearner w(other, tempData, D, labels);
    w.Classify(tempData, predictedLabels);

    // Now, start calculation of alpha(t)

    // building a helper rowvector, mispredict to help in calculations.
    // this stores the value of Yi(l)*ht(xi,l)
    
    // first calculate error: 
    for(j = 0;j < predictedLabels.n_cols; j++)
    {
      if (predictedLabels(j) != labels(j))
      {  
        mispredict(j) = -predictedLabels(j);
        countMP++;
      }
      else
        mispredict(j) = predictedLabels(j);
    }
    et = ((double) countMP / predictedLabels.n_cols);

    if (et < 0.5)
    {
      // begin calculation of rt

      // for (j = 0;j < predictedLabels.n_cols; j++)
      //   rt +=(D(j) * mispredict(j));

      // end calculation of rt

      // alphat = 0.5 * log((1 + rt) / (1 - rt));

      alphat = 0.5 * log((1 - et) / et);  

      // end calculation of alphat
      
      // now start modifying weights

      for (j = 0;j < mispredict.n_cols; j++)
      {
        // we calculate zt, the normalization constant
        zt += D(j) * exp(-1 * alphat * (mispredict(j) / predictedLabels(j)));
        D(j) = D(j) * exp(-1 * alphat * (mispredict(j) / predictedLabels(j)));

        // adding to the matrix of FinalHypothesis 
        if (mispredict(j) == predictedLabels(j)) // if correct prediction
          sumFinalH(j, mispredict(j)) += alphat;
      }
      // normalization of D

      D = D / zt;
    }
  }

  // build a strong hypothesis from a weighted combination of these weak hypotheses.
  
  // This step of storing it in a temporary row vector can be improved upon.
  arma::rowvec tempSumFinalH;

  for (i = 0;i < sumFinalH.n_rows; i++)
  {
    tempSumFinalH = sumFinalH.row(i);
    tempSumFinalH.max(max_index);
    finalH(i) = max_index;
  }
}

} // namespace adaboost
} // namespace mlpack
#endif