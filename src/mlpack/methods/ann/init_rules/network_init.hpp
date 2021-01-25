/**
 * @file methods/ann/init_rules/network_init.hpp
 * @author Marcus Edel
 *
 * Intialization of a given network with a given initialization rule
 * e.g. GaussianInitialization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_NETWORK_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_NETWORK_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

#include "init_rules_traits.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize the network with the given initialization
 * rule.
 */
template<typename InitializationRuleType>
class NetworkInitialization
{
 public:
  /**
   * Use the given initialization rule to initialize the specified network.
   *
   * @param initializeRule Rule to initialize the given network.
   */
  NetworkInitialization(
      const InitializationRuleType& initializeRule = InitializationRuleType()) :
      initializeRule(initializeRule)
  {
    // Nothing to do here.
  }

  /**
   * Initialize the specified network and store the results in the given
   * parameter.
   *
   * @param network Network that should be initialized.
   * @param parameter The network parameter.
   * @param parameterOffset Offset for network paramater, default 0.
   */
  template <typename eT>
  void Initialize(const std::vector<Layer<arma::mat, arma::mat>*>& network,
                  arma::Mat<eT>& parameter, size_t parameterOffset = 0)
  {
    // Determine the number of parameter/weights of the given network.
    if (parameter.is_empty())
    {
      size_t weights = 0;
      for (size_t i = 0; i < network.size(); ++i)
        weights += ModelParameterSize(network[i]);

      parameter.set_size(weights, 1);
    }

    // Initialize the network layer by layer or the complete network.
    if (ann::InitTraits<InitializationRuleType>::UseLayer)
    {
      for (size_t i = 0, offset = parameterOffset; i < network.size(); ++i)
      {
        // Initialize the layer with the specified parameter/weight
        // initialization rule.
        const size_t weight = ModelParameterSize(network[i]);
        arma::Mat<eT> tmp = arma::mat(parameter.memptr() + offset,
            weight, 1, false, false);
        initializeRule.Initialize(tmp, tmp.n_elem, 1);

        // Increase the parameter/weight offset for the next layer.
        offset += weight;
      }
    }
    else
    {
      initializeRule.Initialize(parameter, parameter.n_elem, 1);
    }

    // Note: We can't merge the for loop into the for loop above because
    // ModelParameterSize() also sets the parameter/weights of the inner
    // modules. Inner Modules are held by the parent module e.g. the concat
    // module can hold various other modules.
    size_t offset = parameterOffset;
    for (size_t i = 0; i < network.size(); ++i)
      offset += ModelParameterSize(network[i], offset, parameter);
  }

 private:
  /**
   * Get the size of the weights of the given layer including all sub-layer.
   *
   * @tparam LayerType The type of the layer to get the weight size for.
   * @param layer The layer to get the weight size for.
   * @return The weight size of the given layer and sub-layer.
   */
  template<typename LayerType>
  size_t ModelParameterSize(const LayerType& layer)
  {
    size_t size = 0;

    if (layer->Parameters().n_elem > 0)
      size = layer->Parameters().n_elem;

    if (layer->Model().size() > 0)
    {
      for (size_t i = 0; i < layer->Model().size(); ++i)
        size += ModelParameterSize(layer->Model()[i]);
    }

    return size;
  }

  /**
   * Assign portion of the given parameter matrix to the layer/sub-layer.
   *
   * @tparam LayerType The type of the layer to assign the portion of the
   *     parameter matrix.
   * @param layer The layer to assign the portion of the parameter matrix.
   * @param parameter The complete parameter matrix that will be assigned to
   *     the layer/sub-layer.
   * @return The size of the assigned layer/sub-layer parameter.
   */
  template<typename LayerType>
  size_t ModelParameterSize(const LayerType& layer,
                            size_t offset, arma::mat& parameter)
  {
    size_t size = 0;
    if (layer->Parameters().n_elem > 0)
    {
      layer->Parameters() = arma::mat(parameter.memptr() + offset,
          layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);
      size = layer->Parameters().n_elem;
      layer->Reset();
    }

    if (layer->Model().size() > 0)
    {
      for (size_t i = 0; i < layer->Model().size(); ++i)
        size += ModelParameterSize(layer->Model()[i], offset + size, parameter);
    }

    return size;
  }

  //! Instantiated InitializationRule object for initializing the network
  //! parameter.
  InitializationRuleType initializeRule;
}; // class NetworkInitialization

} // namespace ann
} // namespace mlpack

#endif
