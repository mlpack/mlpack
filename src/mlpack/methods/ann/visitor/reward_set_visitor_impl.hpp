/**
 * @file reward_set_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Reward() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "reward_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! RewardSetVisitor visitor class.
inline RewardSetVisitor::RewardSetVisitor(const double reward) : reward(reward)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void RewardSetVisitor::operator()(LayerType* layer) const
{
  LayerReward(layer);
}

template<typename T>
inline typename std::enable_if<
    HasRewardCheck<T, double&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* layer) const
{
  layer->Reward() = reward;

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(RewardSetVisitor(reward),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    !HasRewardCheck<T, double&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(RewardSetVisitor(reward),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    HasRewardCheck<T, double&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* layer) const
{
  layer->Reward() = reward;
}

template<typename T>
inline typename std::enable_if<
    !HasRewardCheck<T, double&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
RewardSetVisitor::LayerReward(T* /* input */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
