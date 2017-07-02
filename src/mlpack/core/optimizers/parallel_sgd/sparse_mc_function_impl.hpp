/**
 * @file sparse_mc_function_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of the sparse matrix factorization example loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_MC_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_MC_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_mc_function.hpp"

SparseMCLossFunction::SparseMCLossFunction(const arma::uvec& rows,
                                           const arma::uvec& cols,
                                           const arma::vec& ratings,
                                           double mu,
                                           size_t rank) :
  rows(rows), cols(cols), ratings(ratings), mu(mu), rank(rank)
{
  CalculateStatistics();
}

SparseMCLossFunction::SparseMCLossFunction(const arma::sp_mat& dataset,
                                           double mu,
                                           size_t rank) : mu(mu), rank(rank)
{
  // Extract the relevant data from the sparse matrix representation.
  std::vector<double> instance_rows, instance_cols, instance_ratings;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (auto cur = dataset.begin_col(i); cur != dataset.end_col(i); ++cur)
    {
        instance_cols.push_back(i);
        instance_rows.push_back(cur.row());
        instance_ratings.push_back(*cur);
    }
  }
  // Store the data in the object state.
  rows = arma::uvec(instance_rows);
  cols = arma::uvec(instance_cols);
  ratings = arma::vec(instance_ratings);
  CalculateStatistics();
}

void SparseMCLossFunction::CalculateStatistics()
{
  // Take one pass over the data to aggregate statistics.
  numCols = arma::max(cols) + 1;
  numRows = arma::max(rows) + 1;
  // Initialize the statistics aggregate structure.
  colCnt = arma::uvec(numCols, arma::fill::zeros);
  rowCnt = arma::uvec(numRows, arma::fill::zeros);
  // Go through the data and calculate the required frequencies.
  for (size_t i = 0; i < rows.n_elem; ++i)
  {
    rowCnt(rows(i))++;
    colCnt(cols(i))++;
  }
  meanRating = arma::mean(ratings);
}

double SparseMCLossFunction::Evaluate(const arma::mat& weights, size_t id)
{
  // The decision variable is expected to be stored as follows.
  // The first numRows columns have the first factor matrix, the next numCols
  // columns have the second factor matrix. The decision variable matrix is
  // thus of size (numRows + numCols) x rank.

  size_t colId = numRows + cols(id);
  size_t rowId = rows(id);

  float error = arma::dot(weights.col(rowId), weights.col(colId)) +
      meanRating - ratings(id);
  float loss = error * error;
  if (rowCnt(rows(id)) > 1)
    loss += mu * arma::norm(weights.col(rowId)) / (2 * (rowCnt(rows(id)) - 1));
  if (colCnt(cols(id)) > 1)
    loss += mu * arma::norm(weights.col(colId)) / (2 * (colCnt(cols(id)) - 1));
  return loss;
}

void SparseMCLossFunction::Gradient(const arma::mat& weights, size_t id,
                                    arma::sp_mat& gradient)
{
  // Index of the column corresponding to the row and column of the current
  // example in the decision variable.
  size_t colId = numRows + cols(id);
  size_t rowId = rows(id);

  gradient = arma::sp_mat(numRows + numCols, rank);

  double error = arma::dot(weights.col(rowId), weights.col(colId))
      + meanRating - ratings(id);

  // Calculate gradient for the first factor.
  if (rowCnt(rows(id)) > 1)
    gradient.col(rowId) = (mu / (rowCnt(rows(id)) - 1)) * weights.col(rowId);

  gradient.col(rowId) += error * weights.col(colId);

  // Calculate gradient for the second factor.
  if (colCnt(cols(id)) > 1)
    gradient.col(colId) = (mu / (colCnt(cols(id)) - 1)) * weights.col(colId);

  gradient.col(colId) += error * weights.col(rowId);
}

size_t SparseMCLossFunction::NumFunctions()
{
  return rows.n_elem;
}

#endif
