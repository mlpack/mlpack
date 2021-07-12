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
  /**
   * Returns the initial predition for gradient boosting.
   */
  template<typename VecType>
  typename VecType::elem_type InitialPrediction(const VecType& values)
  {
    return arma::accu(values) / (typename VecType::elem_type) values.n_elem;
  }

  /**
   * Returns the first order gradient of the loss function with respect to the
   * values.
   *
   * This is primarily used in calculating the residuals and split gain for the
   * gradient boosted trees.
   *
   * @tparam T The type of input data. This can be both a vector or a scalar.
   * @param observed The true observed values.
   * @param values The values with respect to which the gradient will be
   *     calculated.
   */
  template<typename T>
  T Gradients(const T& observed, const T& values)
  {
    return - (observed - values);
  }

  /**
   * Returns the second order gradient of the loss function with respect to the
   * values. This is used only for vectors.
   */
  template<typename VecType,
           class = std::enable_if_t<VecType::is_col || VecType::is_row>>
  VecType Hessians(const VecType& /* observed */, const VecType& values)
  {
    VecType h(values.n_elem, arma::fill::ones);
    return h;
  }

  /**
   * Returns the pseudo residuals of the predictions.
   * This is equal to the negative gradient of the loss function with respect
   * to the predicted values f.
   *
   * @param observed The true observed values.
   * @param f The prediction at the current step of boosting.
   */
  template<typename VecType>
  VecType Residuals(const VecType& observed, const VecType& f)
  {
    return - Gradients(observed, f);
  }

  /**
   * Returns the output value for the leaf in the tree.
   */
  template<typename VecType>
  typename VecType::elem_type
  OutputValue(const VecType& gradients, const VecType& hessians,
              const double lambda)
  {
    return - arma::accu(gradients) / (arma::accu(hessians) + lambda);
  }

  /**
   * Calculates the similarity score for evaluating the splits.
   */
  template<typename VecType>
  double SimilarityScore(const VecType& observed, const VecType& residuals,
      const size_t begin, const size_t end, const double lambda)
  {
    VecType gradients = Gradients(observed.subvec(begin, end),
        residuals.subvec(begin, end));
    VecType hessians = Hessians(observed.subvec(begin, end),
        residuals.subvec(begin, end));

    return std::pow(arma::accu(gradients), 2) /
        (arma::accu(hessians) + lambda);
  }
};

} // namespace ensemble
} // namespace mlpack

#endif
