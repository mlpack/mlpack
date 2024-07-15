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
  SSELoss() : alpha(0), lambda(0) { /* Nothing to do. */}

  SSELoss(const double alpha, const double lambda):
      alpha(alpha), lambda(lambda)
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

    return accu(values) / (typename VecType::elem_type) values.n_elem;
  }

  /**
   * Returns the output value for the leaf in the tree.
   */
  template<typename MatType, typename WeightVecType>
  double OutputLeafValue(const MatType& /* input */,
                         const WeightVecType& /* weights */)
  {
    return -ApplyL1(accu(gradients)) / (accu(hessians) + lambda);
  }

  /**
   * Calculates the gain from begin to end.
   *
   * @param begin The begin index to calculate gain.
   * @param end The end index to calculate gain.
   */
  double Evaluate(const size_t begin, const size_t end)
  {
    return std::pow(ApplyL1(accu(gradients.subvec(begin, end))), 2) /
        (accu(hessians.subvec(begin, end)) + lambda);
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
    gradients = (input.row(1) - input.row(0)).t();
    hessians = arma::vec(input.n_cols, arma::fill::ones);

    return std::pow(ApplyL1(accu(gradients)), 2) / (accu(hessians) + lambda);
  }
 private:
  //! The L1 regularization parameter.
  const double alpha;
  //! The L2 regularization parameter.
  const double lambda;
  //! First order gradients.
  arma::vec gradients;
  //! Second order gradients (hessians).
  arma::vec hessians;

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

} // namespace mlpack

#endif
