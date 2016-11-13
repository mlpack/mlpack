/**
 * @file simple_weight_update.hpp
 * @author Udit Saxena
 *
 * Simple weight update rule for the perceptron.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _MLPACK_METHODS_PERCEPTRON_LEARNING_POLICIES_SIMPLE_WEIGHT_UPDATE_HPP
#define _MLPACK_METHODS_PERCEPTRON_LEARNING_POLICIES_SIMPLE_WEIGHT_UPDATE_HPP

#include <mlpack/core.hpp>

/**
 * This class is used to update the weightVectors matrix according to the simple
 * update rule as discussed by Rosenblatt:
 *
 *  if a vector x has been incorrectly classified by a weight w,
 *  then w = w - x
 *  and  w'= w'+ x
 *
 *  where w' is the weight vector which correctly classifies x.
 */
namespace mlpack {
namespace perceptron {

class SimpleWeightUpdate
{
 public:
  /**
   * This function is called to update the weightVectors matrix.  It decreases
   * the weights of the incorrectly classified class while increasing the weight
   * of the correct class it should have been classified to.
   *
   * @tparam Type of vector (should be an Armadillo vector like arma::vec or
   *      arma::sp_vec or something similar).
   * @param trainingPoint Point that was misclassified.
   * @param weights Matrix of weights.
   * @param biases Vector of biases.
   * @param incorrectClass Index of class that the point was incorrectly
   *      classified as.
   * @param correctClass Index of the true class of the point.
   * @param instanceWeight Weight to be given to this particular point during
   *      training (this is useful for boosting).
   */
  template<typename VecType>
  void UpdateWeights(const VecType& trainingPoint,
                     arma::mat& weights,
                     arma::vec& biases,
                     const size_t incorrectClass,
                     const size_t correctClass,
                     const double instanceWeight = 1.0)
  {
    weights.col(incorrectClass) -= instanceWeight * trainingPoint;
    biases(incorrectClass) -= instanceWeight;

    weights.col(correctClass) += instanceWeight * trainingPoint;
    biases(correctClass) += instanceWeight;
  }
};

} // namespace perceptron
} // namespace mlpack

#endif
