/**
 * @file adaboost.hpp
 * @author Udit Saxena
 *
 * AdaBoost header file
 */

#ifndef _MLPACK_METHODS_ADABOOST_ADABOOST_HPP
#define _MLPACK_METHODS_ADABOOST_ADABOOST_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>
 
namespace mlpack {
namespace adaboost {

template <typename MatType = arma::mat, typename WeakLearner = 
          mlpack::perceptron::Perceptron<> >
class Adaboost 
{
public:
  Adaboost(const MatType& data, const arma::Row<size_t>& labels,
           int iterations, size_t classes, const WeakLearner& other);

  void buildClassificationMatrix(arma::mat& t, const arma::Row<size_t>& l);

  void buildWeightMatrix(const arma::mat& D, arma::rowvec& weights);

}; // class Adaboost

} // namespace adaboost
} // namespace mlpack

#include "adaboost_impl.hpp"

#endif