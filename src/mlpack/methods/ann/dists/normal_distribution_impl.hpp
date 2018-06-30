/**
 * @file normal_distribution.cpp
 * @author Atharva Khandait
 *
 * Implementation of the Normal distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "normal_distribution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename DataType>
NormalDistribution<DataType>::NormalDistribution() :
    applySoftplus(true)
{
  // Nothing to do here.
}

template<typename DataType>
NormalDistribution<DataType>::NormalDistribution(
    const DataType&& mean,
    const DataType&& stdDev) :
    mean(mean),
    stdDev(stdDev)
{
  if (mean.size() != stdDev.size())
  {
    Log::Fatal << "NormalDistribution<>::NormalDistribution(): The sizes of "
        << "the mean and the standard deviation should be equal." << std::endl;
  }
}

template<typename DataType>
NormalDistribution<DataType>::NormalDistribution(
    const DataType&& param,
    const bool applySoftplus) :
    mean(param.submat(param.n_rows / 2, 0, param.n_rows - 1,
        param.n_cols - 1)),
    preStdDev(param.submat(0, 0, param.n_rows / 2 - 1, param.n_cols - 1)),
    applySoftplus(applySoftplus)
{
  if (param.n_rows % 2 != 0)
  {
    Log::Fatal << "NormalDistribution<>::NormalDistribution(): The number of "
        << "rows of param matrix should be even." << std::endl;
  }

  if (applySoftplus)
    SoftplusFunction::Fn(preStdDev, stdDev);
  else
    stdDev = preStdDev;
}

template<typename DataType>
DataType NormalDistribution<DataType>::Sample() const
{
  return stdDev % arma::randn<DataType>(mean.n_rows, mean.n_cols) + mean;
}

template<typename DataType>
double NormalDistribution<DataType>::LogProbability(
    const DataType&& observation) const
{
  if (observation.size() != mean.size())
  {
    Log::Fatal << "NormalDistribution<>::NormalDistribution(): The size of the"
        << "observation should be equal to the sizes of the mean and standard"
        << "deviation." << std::endl;
  }

  return -0.5 * (arma::accu(2 * arma::log(stdDev) + arma::pow(
      (mean - observation) / stdDev, 2) + log2pi));
}

template<typename DataType>
void NormalDistribution<DataType>::LogProbBackward(
    const DataType&& observation, DataType&& output) const
{
  if (!applySoftplus)
  {
    output = -0.5 * join_cols((2 / stdDev - 2 *
        arma::pow(mean - observation, 2) / arma::pow(stdDev, 3)), 2 *
        (mean - observation) / arma::pow(stdDev, 2));
  }
  else
  {
    SoftplusFunction::Deriv(preStdDev, output);
    output = -0.5 * join_cols((2 / stdDev - 2 *
        arma::pow(mean - observation, 2) / arma::pow(stdDev, 3)) % output,
        2 * (mean - observation) / arma::pow(stdDev, 2));
  }
}

} // namespace ann
} // namespace mlpack

#endif
