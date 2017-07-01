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

SparseMCLossFunction::SparseMCLossFunction(arma::uvec &rows, arma::uvec &cols,
                                           arma::vec &ratings, size_t rank) :
  rows(rows), cols(cols), ratings(ratings), rank(rank)
{ /* Nothing to do */ }

SparseMCLossFunction::SparseMCLossFunction(arma::sp_mat &dataset,
                                           size_t rank) : rank(rank)
{
  // Extract the relevant data from the sparse matrix representation.
  std::vector<double> instance_rows, instance_cols, instance_ratings;
  for(size_t i = 0; i < dataset.n_cols; ++i)
  {
    for(auto cur = dataset.begin_col(i); cur != dataset.end_col(i); ++cur)
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
}

void SparseMCLossFunction::CalculateStatistics()
{
  // Take one pass over the data to aggregate statistics.
  size_t n_cols = arma::max(cols);
  size_t n_rows = arma::max(rows);
  // Initialize the statistics aggregate structure.
  colCnt = arma::uvec(n_cols, arma::fill::zeros);
  rowCnt = arma::uvec(n_rows, arma::fill::zeros);
  // Go through the data and calculate the required frequency.
  for(size_t i = 0; i < rows.n_elem; ++i)
  {
    rowCnt[rows[i]]++;
    colCnt[cols[i]]++;
  }
  meanRating = arma::mean(ratings);
}

double SparseMCLossFunction::Evaluate(const arma::mat &weights, size_t id)
{
  // The decision variable is expected to be stored as follows.
  // The first numRows columns have the first factor matrix, the next numCols
  // columns have the second factor matrix. The decision variable matrix is
  // thus of size (numRows + numCols) x rank.
  float error = arma::dot(weights.col(rows(id)), weights.col(numRows + id)) +
      meanRating - ratings(id);
  return error * error;
}

void SparseMCLossFunction::Gradient(const arma::mat &weights, size_t id,
                                    arma::sp_mat &gradient)
{
  gradient = arma::sp_mat(numRows + numCols, rank);
  // We only need to alter the relevant row and column in the decision variable.
  double error = arma::Dot(weights.col(rows(id)), weights.col(numRows + id))
      + meanRating - ratings(id);
}

size_t SparseMCLossFunction::NumFunctions()
{
  return rows.n_elem;
}

#endif
