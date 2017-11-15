/**
 * @file vr_class_reward_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the VRClassReward class, which implements the variance
 * reduced classification reinforcement layer.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_IMPL_HPP

// In case it hasn't yet been included.
#include "vr_class_reward.hpp"

#include "../visitor/reward_set_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
VRClassReward<InputDataType, OutputDataType>::VRClassReward(
    const double scale,
    const bool sizeAverage) :
    scale(scale),
    sizeAverage(sizeAverage)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double VRClassReward<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  double output = 0;
  for (size_t i = 0; i < input.n_cols - 1; ++i)
  {
    size_t currentTarget = target(i) - 1;
    Log::Assert(currentTarget >= 0 && currentTarget < input.n_rows,
        "Target class out of range.");

    output -= input(currentTarget, i);
  }

  reward = 0;
  arma::uword index = 0;

  for (size_t i = 0; i < input.n_cols - 1; i++)
  {
    input.unsafe_col(i).max(index);
    reward = ((index + 1) == target(i)) * scale;
  }

  if (sizeAverage)
  {
    return output - reward / (input.n_cols - 1);
  }

  return output - reward;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void VRClassReward<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  output = arma::zeros<OutputType>(input.n_rows, input.n_cols);
  for (size_t i = 0; i < (input.n_cols - 1); ++i)
  {
    size_t currentTarget = target(i) - 1;
    Log::Assert(currentTarget >= 0 && currentTarget < input.n_rows,
        "Target class out of range.");

    output(currentTarget, i) = -1;
  }

  double vrReward = reward - input(0, 1);
  if (sizeAverage)
  {
    vrReward /= input.n_cols - 1;
  }

  const double norm = sizeAverage ? 2.0 / (input.n_cols - 1) : 2.0;

  output(0, 1) = norm * (input(0, 1) - reward);
  boost::apply_visitor(RewardSetVisitor(vrReward), network.back());
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void VRClassReward<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
