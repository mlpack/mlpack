/**
 * @file sparse_mc_function.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of the hinge loss function for training a sparse SVM with the
 * parallel SGD algorithm
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_MC_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_MC_HPP
#include <mlpack/prereqs.hpp>

/**
 * An implementation of the matrix completion example from HOGWILD!, based on
 * empirical risk minimization in a sparse setting.
 */

class SparseMCLossFunction{
 public:
  //! Nothing to do for the default constructor.
  SparseMCLossFunction() = default;

  /**
   * Member initialization constructor.
   *
   * @param rows The row indices of the data points.
   * @param cols The column indices of the data points.
   * @param ratings The ratings of the data points.
   * @param mu The regularization parameter.
   * @param rank The width of the first factor.
   */
  SparseMCLossFunction(const arma::uvec& rows, const arma::uvec& cols,
      const arma::vec& ratings, double mu, size_t rank);

  /**
   * Special initialization constructor.
   *
   * @param dataset The sparse matrix containing the datapoints.
   * @param mu The regularization parameter.
   * @param rank The width of the first factor.
   */
  SparseMCLossFunction(const arma::sp_mat& dataset, double mu, size_t rank);

  /**
   * Evaluate the squared error function with the given parameters at the id-th
   * data point.
   *
   * @param weights The decision variable at which the function is to be
   *     evaluated.
   * @param id Index of point to use for objective function evaluation.
   * @return The value of the loss function at the given parameter.
   */
  double Evaluate(const arma::mat& weights, size_t id);

  /**
   * Evaluate the gradient of the squared error with the given parameters.
   *
   * @tparam GradType The type of the gradient parameter.
   * @param weights The decision variable at which the gradient is to be
   *     evaluated.
   * @param id Index of point to use for objective function evaluation.
   * @param gradient Out param for the gradient.
   */
  template <typename GradType>
  void Gradient(const arma::mat& weights, size_t id, GradType& gradient);

  //! Get the height of the sparse matrix.
  size_t NumRows() const { return numRows; }
  //! Modify the height of the sparse matrix.
  size_t& NumRows() { return numRows; }

  //! Get the width of the sparse matrix.
  size_t NumCols() const { return numCols; }
  //! Modify the width of the sparse matrix.
  size_t& NumCols() { return numCols; }

  //! Get the regularization parameter.
  double Mu() const { return mu; }
  //! Modify the regularization parameter.
  double& Mu() { return mu; }

  //! Get the rank.
  size_t Rank() const { return rank; }
  //! Modify the rank.
  size_t& Rank() { return rank; }

  //! Return the number of functions.
  size_t NumFunctions();

 private:
  //! Calculate the frequency tables and mean rating before calling Evaluate
  //! or Gradient.
  void CalculateStatistics();

  //! The row index of the datapoints.
  arma::uvec rows;

  //! The column index of the datapoints.
  arma::uvec cols;

  //! The rating of the datapoints.
  arma::vec ratings;

  //! The frequency of the columns.
  arma::uvec colCnt;

  //! The frequency of the rows.
  arma::uvec rowCnt;

  //! The regularization parameter.
  double mu;

  //! The height of the sparse matrix
  size_t numRows;

  //! The width of the sparse matrix
  size_t numCols;

  //! The width of the first factor.
  size_t rank;
};

// Include implementation
#include "sparse_mc_function_impl.hpp"

#endif
