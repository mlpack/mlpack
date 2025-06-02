/**
 * @file methods/ann/loss_functions/vr_class_reward_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the VRClassRewardType class, which implements the variance
 * reduced classification reinforcement layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_VR_CLASS_REWARD_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_VR_CLASS_REWARD_IMPL_HPP

// In case it hasn't yet been included.
#include "vr_class_reward.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename MatType>
VRClassRewardType<MatType>::VRClassRewardType(
    const double scale,
    const bool sizeAverage) :
    scale(scale),
    sizeAverage(sizeAverage),
    reward(0)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type VRClassRewardType<MatType>::Forward(
    const MatType& input, const MatType& target)
{
  double output = 0;
  for (size_t i = 0; i < input.n_cols - 1; ++i)
  {
    const size_t currentTarget = target(i);
    Log::Assert(currentTarget < input.n_rows, "Target class out of range.");

    output -= input(currentTarget, i);
  }

  reward = 0;
  arma::uword index = 0;

  for (size_t i = 0; i < input.n_cols - 1; ++i)
  {
    index = input.unsafe_col(i).index_max();
    reward = (index == target(i)) * scale;
  }

  if (sizeAverage)
  {
    return output - reward / (input.n_cols - 1);
  }

  return output - reward;
}

template<typename MatType>
void VRClassRewardType<MatType>::Backward(
    const MatType& input,
    const MatType& target,
    MatType& output)
{
  output = zeros<MatType>(input.n_rows, input.n_cols);
  for (size_t i = 0; i < (input.n_cols - 1); ++i)
  {
    const size_t currentTarget = target(i);
    Log::Assert(currentTarget < input.n_rows, "Target class out of range.");

    output(currentTarget, i) = -1;
  }

  double vrReward = reward - input(0, 1);
  if (sizeAverage)
  {
    vrReward /= input.n_cols - 1;
  }

  const double norm = sizeAverage ? 2.0 / (input.n_cols - 1) : 2.0;

  output(0, 1) = norm * (input(0, 1) - reward);
  network.back()->Reward() = vrReward;
}

template<typename MatType>
template<typename Archive>
void VRClassRewardType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(scale));
  ar(CEREAL_NVP(sizeAverage));
  ar(CEREAL_NVP(reward));
  ar(CEREAL_NVP(network));
}

} // namespace mlpack

#endif
