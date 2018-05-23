/**
 * @file reward_clipping.hpp
 * @author Chenna Keshava B S
 *
 * Class for implementiong Reward clipping 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_SRC_MLPACK_METHODS_REINFORCEMENT_LEARNING_REWARD_CLIPPING_HPP

#define MLPACK_SRC_MLPACK_METHODS_REINFORCEMENT_LEARNING_REWARD_CLIPPING_HPP


namespace mlpack {
namespace rl {

/**
 * Interface for clipping the reward to some value between the specified maximum and minimum value
 * (Clipping here is implemented as
 * \f$ g_{\text{clipped}} = \max(g_{\text{min}}, \min(g_{\text{min}}, g))) \f$.)
 */

class RewardClipping
{
 public:
  /**
   * Constructor for creating a RewardClipping instance.
   *
   * @param minReward Minimum possible value of reward
   * @param maxReward Maximum possible value of reward
   */
  RewardClipping(const double minReward,
                   const double maxReward) :
    minReward(minReward),
    maxReward(maxReward)
  {
    // Nothing to do here
  }

  /**
   * Clipping step - First the reward is clipped using Armadillo's clamp function.
   * Then I am returning a reference to the clipped reward.
   *
   * @param reward The reward value
   */
  double GetReward(double reward) 
  {
    // First, clip the reward.
    
    if (reward > maxReward)
        clippedReward = maxReward;
    else if (reward < minReward)
        clippedReward = minReward;
    else
      clippedReward = reward;

    return clippedReward;
  }

  

  //! Get the minimum reward value.
  double MinReward() const { return minReward; }
  //! Modify the minimum reward value.
  double& MinReward() { return minReward; }

  //! Get the maximum reward value.
  double MaxReward() const { return maxReward; }
  //! Modify the maximum reward value.
  double& MaxReward() { return maxReward; }

 private:
  //! Minimum possible value of reward.
  double minReward;

  //! Maximum possible value of reward.
  double maxReward;
  double clippedReward;
};

} // namespace rl
} // namespace mlpack

#endif
