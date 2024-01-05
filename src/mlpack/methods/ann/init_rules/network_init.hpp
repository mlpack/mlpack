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

/**
 * Create a temporary matrix with a specific offset, this is only necessary
 * since we have two  backend libraries to support. The first function
 * overload is called when the MatType == arma.
 *
 * @param tmp The returned temporary matrix.
 * @param Mat The original matrix we are taking part from it.
 * @param offset The Start point of the tmp matrix.
 * @param numRows The number of rows of the tmp matrix.
 * @param numCols The numbers or cols of the tmp matrix.
 */
template<typename MatType,
         typename = typename std::enable_if<
         !IsCootMatType<MatType>::value>::type>
void MakeTmp(MatType& tmp,
             const MatType& Mat,
             const size_t offset,
             const size_t numRows,
             const size_t numCols)
{
  tmp = MatType(Mat.memptr() + offset, numRows, numCols, false,
      false);
}

#ifdef MLPACK_HAS_COOT
/**
 * Create a temporary matrix with a specific offset, this is only necessary
 * since we have two  backend libraries to support. The first function
 * overload is called when the MatType == coot.
 *
 * @param tmp The returned temporary matrix.
 * @param Mat The original matrix we are taking part from it.
 * @param offset The Start point of the tmp matrix.
 * @param numRows The number of rows of the tmp matrix.
 * @param numCols The numbers or cols of the tmp matrix.
 */
template<typename MatType,
         typename = typename std::enable_if<
         IsCootMatType<MatType>::value>::type>
void MakeTmp(MatType& tmp,
             const MatType& Mat,
             const size_t offset,
             const size_t numRows,
             const size_t numCols)
{
  tmp = MatType(Mat.get_dev_mem() + offset, numRows, numCols, false,
      false);
}

#endif

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
  template <typename MatType>
  void Initialize(const std::vector<Layer<MatType>*>& network,
                  MatType& parameters,
                  size_t parameterOffset = 0)
  {
    // Determine the total number of parameters/weights of the given network.
    if (parameters.is_empty())
    {
      size_t weights = 0;
      for (size_t i = 0; i < network.size(); ++i)
        weights += network[i]->WeightSize();

      parameters.set_size(weights, 1);
    }

    // Initialize the network layer by layer or the complete network.
    if (InitTraits<InitializationRuleType>::UseLayer)
    {
      for (size_t i = 0, offset = parameterOffset; i < network.size(); ++i)
      {
        // Initialize the layer with the specified parameter/weight
        // initialization rule.
        const size_t weight = network[i]->WeightSize();
        MatType tmp;
        MakeTmp(tmp, parameters, offset, weight, 1);
        initializeRule.Initialize(tmp, tmp.n_elem, 1);

        // Increase the parameter/weight offset for the next layer.
        offset += weight;
      }
    }
    else
    {
      initializeRule.Initialize(parameters, parameters.n_elem, 1);
    }
  }

 private:
  //! Instantiated InitializationRule object for initializing the network
  //! parameter.
  InitializationRuleType initializeRule;
}; // class NetworkInitialization

} // namespace mlpack

#endif
