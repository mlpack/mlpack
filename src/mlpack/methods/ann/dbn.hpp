/**
 * @file methods/ann/dbn.hpp
 * @author Himanshu Pathak
 *
 * Definition of the DBN class, which implements deep belief neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FFN_HPP
#define MLPACK_METHODS_ANN_FFN_HPP

#include <mlpack/prereqs.hpp>

#include "visitor/delete_visitor.hpp"
#include "visitor/delta_visitor.hpp"
#include "visitor/output_height_visitor.hpp"
#include "visitor/output_parameter_visitor.hpp"
#include "visitor/output_width_visitor.hpp"
#include "visitor/reset_visitor.hpp"
#include "visitor/weight_size_visitor.hpp"
#include "visitor/copy_visitor.hpp"
#include "visitor/loss_visitor.hpp"

#include "init_rules/network_init.hpp"

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <ensmallen.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of a deep belief network.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam CustomLayers Any set of custom layers that could be a part of the
 *         feed forward network.
 */
template<
  typename OutputLayerType = NegativeLogLikelihood<>,
  typename InitializationRuleType = RandomInitialization,
  typename... CustomLayers
>
class DBN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = DBN<OutputLayerType, InitializationRuleType>;

  /**
   * Create the DBN object.
   *
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  DBN(OutputLayerType outputLayer = OutputLayerType(),
      InitializationRuleType initializeRule = InitializationRuleType());

  //! Copy constructor.
  DBN(const DBN&);

  //! Move constructor.
  DBN(DBN&&);

  //! Copy/move assignment operator.
  DBN& operator = (DBN);

  //! Destructor to release allocated memory.
  ~DBN();
}