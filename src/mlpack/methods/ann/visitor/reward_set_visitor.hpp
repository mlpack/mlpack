/**
 * @file reward_set_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Reward() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * RewardSetVisitor set the reward parameter given the reward value.
 */
class RewardSetVisitor : public boost::static_visitor<void>
{
 public:
  //! Set the reward parameter given the reward value.
  RewardSetVisitor(const double reward);

  //! Set the reward parameter.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The reward value.
  const double reward;

  //! Set the deterministic parameter if the module implements the
  //! Deterministic() and Model() function.
  template<typename T>
  typename std::enable_if<
      HasRewardCheck<T, double&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;

  //! Set the deterministic parameter if the module implements the
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRewardCheck<T, double&(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;

  //! Set the deterministic parameter if the module implements the
  //! Deterministic() function.
  template<typename T>
  typename std::enable_if<
      HasRewardCheck<T, double&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;

  //! Do not set the deterministic parameter if the module doesn't implement the
  //! Deterministic() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRewardCheck<T, double&(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  LayerReward(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "reward_set_visitor_impl.hpp"

#endif
