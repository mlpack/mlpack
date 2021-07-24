/**
 * @file methods/xgboost/loss_functions/sse_loss.hpp
 * @author Rishabh Garg
 *
 * The sum of squared error loss class, which is a loss funtion for gradient
 * xgboost based decision trees.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_SSE_LOSS_HPP
#define MLPACK_METHODS_XGBOOST_LOSS_FUNCTIONS_SSE_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ensemble {

/**
 * The SSE (Sum of Squared Errors) loss is a loss function to measure the
 * quality of prediction of response values present in the node of each
 * xgboost tree. It is also a good measure to compare the spread of two
 * distributions. We will try to minimize this value while training.
 *
 * Loss = 1 / 2 * (Observed - Predicted)^2
 */
class SSELoss
{
 public:
  // Default constructor---No regularization.
  SSELoss(): alpha(0), lambda(0), minChildWeight(1) { /* Nothing to do. */}

  SSELoss(const double alpha, const double lambda,
      const double minChildWeight) :
      alpha(alpha), lambda(lambda), minChildWeight(minChildWeight)
  {
    // Nothing to do.
  }

  /**
   * Returns the initial predition for gradient boosting.
   */
  template<typename VecType>
  typename VecType::elem_type InitialPrediction(const VecType& values)
  {
    // Sanity check for empty vector.
    if (values.n_elem == 0)
      return 0;

    return arma::accu(values) / (typename VecType::elem_type) values.n_elem;
  }

  /**
   * Returns the output value for the leaf in the tree.
   */
  template<typename MatType, typename WeightVecType>
  double OutputLeafValue(const MatType& /* input */,
                         const WeightVecType& /* weights */)
  {
    return -ApplyL1(arma::accu(gradients)) / (arma::accu(hessians) + lambda);
  }

  /**
   * Calculates the gain from begin to end.
   *
   * @param begin The begin index to calculate gain.
   * @param end The end index to calculate gain.
   */
  double Evaluate(const size_t begin, const size_t end)
  {
    return std::pow(ApplyL1(arma::accu(gradients.subvec(begin, end))), 2) /
        (arma::accu(hessians.subvec(begin, end)) + lambda);
  }

  /**
   * Calculates the gain of the node before splitting. It also initializes the
   * gradients and hessians used later for finding split.
   * UseWeights and weights are ignored here. These are just to make the API
   * consistent.
   *
   * @param input This is a 2D matrix. The first row stores the true observed
   *    values and the second row stores the prediction at the current step
   *    of boosting.
   */
  template<bool UseWeights, typename MatType, typename WeightVecType>
  double Evaluate(const MatType& input, const WeightVecType& /* weights */)
  {
    // Calculate gradients and hessians.
    gradients = input.row(1) - input.row(0);
    hessians = arma::vec(input.n_cols, arma::fill::ones);

    // Calculates total sum of gradients and hessians.
    gTotal = arma::accu(gradients);
    hTotal = arma::accu(hessians);

    return std::pow(ApplyL1(gTotal), 2) / (hTotal + lambda);
  }

  /**
   * Sorts the gradients and hessians according to the sorted order of the
   * feature.
   *
   * @param sortedIndices Vector of indices according to the sorted order of
   *      the feature.
   * @param sortedGradients Vector to store the sorted gradients.
   * @param sortedHessians Vector to store the sorted hessians.
   */
  void sortGradAndHess(const arma::uvec& sortedIndices,
                       arma::vec& sortedGradients,
                       arma::vec& sortedHessians)
  {
    for (size_t i = 0; i < sortedIndices.n_elem; ++i)
    {
      sortedGradients[i] = gradients[sortedIndices[i]];
      sortedHessians[i] = hessians[sortedIndices[i]];
    }
  }

  /**
   * Calculates the sum of hessians and gradients for the left child.
   *
   * @return The start index for the iteration to find the best split.
   */
  size_t BinaryScanInitialize(const arma::vec& sortedGradients,
                              const arma::vec& sortedHessians)
  {
    size_t index = 0;
    // Initializing the data members.
    gLeft = 0;
    hLeft = 0;
    while (index < sortedHessians.n_elem - 1 &&
           hLeft + sortedHessians[index] < minChildWeight)
    {
      gLeft += sortedGradients[index];
      hLeft += sortedHessians[index++];
    }
    return index;
  }

  /**
   * Steps through the current index and update the statistics. While taking
   * the step, it ensures that the weight of right child is higher than the
   * minChildWeight.
   */
  void BinaryStep(const arma::vec& sortedGradients,
                  const arma::vec& sortedHessians,
                  size_t index,
                  bool& endLoop)
  {
    gLeft += sortedGradients[index];
    hLeft += sortedHessians[index];

    // If the total weight of right child is lesser than the minChildWeight,
    // then we need to terminate the loop for iterating over split points.
    if (hTotal - hLeft < minChildWeight)
      endLoop = true;
    return;
  }

  /**
   * Calculates the gains for the current split.
   */
  double Evaluate()
  {
    double leftGain = std::pow(ApplyL1(gLeft), 2) / (hLeft + lambda);
    double rightGain = std::pow(ApplyL1(gTotal - gLeft), 2) /
        (hTotal - hLeft + lambda);

    return leftGain + rightGain;
  }
 private:
  //! The L2 regularization parameter.
  const double lambda;
  //! The L1 regularization parameter.
  const double alpha;
  //! The minimum total weight possible for any child. A higher value of this
  //! parameter reduces overfitting.
  const double minChildWeight;
  //! First order gradients.
  arma::vec gradients;
  //! Second order gradients (hessians).
  arma::vec hessians;
  //! Sum of gradients of left child.
  double gLeft;
  //! Total sum of gradients.
  double gTotal;
  //! Sum of hessians of left child.
  double hLeft;
  //! Total sum of hessians.
  double hTotal;

  //! Applies the L1 regularization.
  double ApplyL1(const double sumGradients)
  {
    if (sumGradients > alpha)
    {
      return sumGradients - alpha;
    }
    else if (sumGradients < - alpha)
    {
      return sumGradients + alpha;
    }
    
    return 0;
  }
};

} // namespace ensemble
} // namespace mlpack

#endif
