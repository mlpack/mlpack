/**
 * @file methods/regularized_svd/regularized_svd_function.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of the RegularizedSVDFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_FUNCTION_SVD_HPP
#define MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_FUNCTION_SVD_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

namespace mlpack {

/**
 * The data is stored in a matrix of type MatType, so that this class can be
 * used with both dense and sparse matrix types.
 *
 * @tparam MatType The matrix type of the dataset.
 */
template <typename MatType = arma::mat>
class RegularizedSVDFunction
{
 public:
  /**
   * Constructor for RegularizedSVDFunction class. The constructor calculates
   * the number of users and items in the passed data. It also randomly
   * initializes the parameter values.
   *
   * @param data Dataset for which SVD is calculated.
   * @param rank Rank used for matrix factorization.
   * @param lambda Regularization parameter used for optimization.
   */
  RegularizedSVDFunction(const MatType& data,
                         const size_t rank,
                         const double lambda);

  /**
   * Shuffle the points in the dataset.  This may be used by optimizers.
   */
  void Shuffle();

  /**
   * Evaluates the cost function over all examples in the data.
   *
   * @param parameters Parameters(user/item matrices) of the decomposition.
   */
  double Evaluate(const arma::mat& parameters) const;

  /**
   * Evaluates the cost function for one training example. Useful for the SGD
   * optimizer abstraction which uses one training example at a time.
   *
   * @param parameters Parameters(user/item matrices) of the decomposition.
   * @param start First index of the training examples to be used.
   * @param batchSize Size of batch to evaluate.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t start,
                  const size_t batchSize = 1) const;

  /**
   * Evaluates the full gradient of the cost function over all the training
   * examples.
   *
   * @param parameters Parameters(user/item matrices) of the decomposition.
   * @param gradient Calculated gradient for the parameters.
   */
  void Gradient(const arma::mat& parameters,
                arma::mat& gradient) const;

  /**
   * Evaluates the gradient of the cost function over one training example.
   * This function is useful for optimizers like SGD. The type of the gradient
   * parameter is a template argument to allow the computation of a sparse
   * gradient.
   *
   * @tparam GradType The type of the gradient out-param.
   * @param parameters Parameters(user/item matrices) of the decomposition.
   * @param start The first index of the training examples to use.
   * @param gradient Calculated gradient for the parameters.
   * @param batchSize Size of batch to calculate gradient for.
   */
  template <typename GradType>
  void Gradient(const arma::mat& parameters,
                const size_t start,
                GradType& gradient,
                const size_t batchSize = 1) const;

  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  //! Return the dataset passed into the constructor.
  const arma::mat& Dataset() const { return data; }

  //! Return the number of training examples. Useful for SGD optimizer.
  size_t NumFunctions() const { return data.n_cols; }

  //! Return the number of users in the data.
  size_t NumUsers() const { return numUsers; }

  //! Return the number of items in the data.
  size_t NumItems() const { return numItems; }

  //! Return the regularization parameters.
  double Lambda() const { return lambda; }

  //! Return the rank used for the factorization.
  size_t Rank() const { return rank; }

 private:
  //! Rating data.  This will be an alias until Shuffle() is called.
  MatType data;
  //! Initial parameter point.
  arma::mat initialPoint;
  //! Rank used for matrix factorization.
  size_t rank;
  //! Regularization parameter for the optimization.
  double lambda;
  //! Number of users in the given dataset.
  size_t numUsers;
  //! Number of items in the given dataset.
  size_t numItems;
};

} // namespace mlpack

namespace ens {

  /**
   * Template specialization for the SGD and parallel SGD optimizer. Used
   * because the gradient affects only a small number of parameters per example,
   * and thus the normal abstraction does not work as fast as we might like it
   * to.
   */
  template <>
  template <>
  inline double StandardSGD::Optimize(
      mlpack::RegularizedSVDFunction<arma::mat>& function,
      arma::mat& parameters);

  template <>
  template <>
  inline double ParallelSGD<ExponentialBackoff>::Optimize(
      mlpack::RegularizedSVDFunction<arma::mat>& function,
      arma::mat& parameters);

} // namespace ens

#include "regularized_svd_function_impl.hpp"

#endif
