/**
 * @file q_network.hpp
 * @author Nishant Kumar
 *
 * This file is consists of the q network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_Q_NETWORK_HPP
#define MLPACK_METHODS_RL_Q_NETWORK_HPP

#include <mlpack/prereqs.hpp>
#include "training_config.hpp"


namespace mlpack {
namespace rl {

template <typename FeedForwardNetworkType>
class QNetwork
{
 public:

  QNetwork()
  {}
 
  QNetwork(FeedForwardNetworkType network) : network(std::move(network))
  { /* Nothing to do here. */ }
  
  void Predict(const arma::colvec& state,
             arma::colvec& actionValue)
  {
    network.Predict(state, actionValue);
  }

  void Forward(arma::colvec& state,
             arma::mat& target)
  {
    network.Forward(state, target);
  }

  void ResetParametersIfEmpty()
  {
    if (network.Parameters().is_empty())
      network.ResetParameters();
  }
 private:
  //! Locally-stored number of examples of each sample.
  FeedForwardNetworkType network;
};


} // namespace rl
} // namespace mlpack

#endif
