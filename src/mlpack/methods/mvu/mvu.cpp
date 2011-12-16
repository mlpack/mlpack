/**
 * @file mvu.cpp
 * @author Ryan Curtin
 *
 * Implementation of the MVU class and its auxiliary objective function class.
 */
#include "mvu.hpp"
#include "mvu_objective_function.hpp"

#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

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
  MVUObjectiveFunction obj(data, newDim, numNeighbors);

  // Set up Augmented Lagrangian method.
  // Memory choice is arbitrary; this needs to be configurable.
  AugLagrangian<MVUObjectiveFunction> aug(obj, 20);

  outputData = obj.GetInitialPoint();
  aug.Optimize(outputData, 0);
}
