/**
 * @file mvu.cpp
 * @author Ryan Curtin
 *
 * Implementation of the MVU class and its auxiliary objective function class.
 *
 * Note: this implementation of MVU does not work.  See #189.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "mvu.hpp"

//#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
#include <mlpack/core/optimizers/sdp/lrsdp.hpp>

#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;
using namespace mlpack::mvu;
using namespace mlpack::optimization;

MVU::MVU(const arma::mat& data) : data(data)
{
  // Nothing to do.
}

void MVU::Unfold(const size_t newDim,
                 const size_t numNeighbors,
                 arma::mat& outputData)
{
  // First we have to choose the output point.  We'll take a linear projection
  // of the data for now (this is probably not a good final solution).
//  outputData = trans(data.rows(0, newDim - 1));
  // Following Nick's idea.
  outputData.randu(data.n_cols, newDim);

  // The number of constraints is the number of nearest neighbors plus one.
  LRSDP<arma::sp_mat> mvuSolver(numNeighbors * data.n_cols + 1, outputData);

  // Set up the objective.  Because we are maximizing the trace of (R R^T),
  // we'll instead state it as min(-I_n * (R R^T)), meaning C() is -I_n.
  mvuSolver.C().eye(data.n_cols, data.n_cols);
  mvuSolver.C() *= -1;

  // Now set up each of the constraints.
  // The first constraint is trace(ones * R * R^T) = 0.
  mvuSolver.B()[0] = 0;
  mvuSolver.A()[0].ones(data.n_cols, data.n_cols);

  // All of our other constraints will be sparse except the first.  So set that
  // vector of modes accordingly.
  mvuSolver.AModes().ones();
  mvuSolver.AModes()[0] = 0;

  // Now all of the other constraints.  We first have to run KNN to get the
  // list of nearest neighbors.
  arma::Mat<size_t> neighbors;
  arma::mat distances;

  KNN knn(data);
  knn.Search(numNeighbors, neighbors, distances);

  // Add each of the other constraints.  They are sparse constraints:
  //   Tr(A_ij K) = d_ij;
  //   A_ij = zeros except for 1 at (i, i), (j, j); -1 at (i, j), (j, i).
  for (size_t i = 0; i < neighbors.n_cols; ++i)
  {
    for (size_t j = 0; j < numNeighbors; ++j)
    {
      // This is the index of the constraint.
      const size_t index = (i * numNeighbors) + j + 1;

      arma::mat& aRef = mvuSolver.A()[index];

      aRef.set_size(3, 4);

      // A_ij(i, i) = 1.
      aRef(0, 0) = i;
      aRef(1, 0) = i;
      aRef(2, 0) = 1;

      // A_ij(i, j) = -1.
      aRef(0, 1) = i;
      aRef(1, 1) = neighbors(j, i);
      aRef(2, 1) = -1;

      // A_ij(j, i) = -1.
      aRef(0, 2) = neighbors(j, i);
      aRef(1, 2) = i;
      aRef(2, 2) = -1;

      // A_ij(j, j) = 1.
      aRef(0, 3) = neighbors(j, i);
      aRef(1, 3) = neighbors(j, i);
      aRef(2, 3) = 1;

      // The constraint b_ij is the distance between these two points.
      mvuSolver.B()[index] = distances(j, i);
    }
  }

  // Now on with the solving.
  double objective = mvuSolver.Optimize(outputData);

  Log::Info << "Final objective is " << objective << "." << std::endl;

  // Revert to original data format.
  outputData = trans(outputData);
}
