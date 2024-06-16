/**
 * @file methods/ann/init_rules/layer_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the LayerInitialization method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_LAYER_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_LAYER_INIT_HPP

#include <mlpack/prereqs.hpp>

#include "init_rules_traits.hpp"

namespace mlpack {

/**
 * This class does nothing and is used to initialize the weight matrix with the
 * weights form the network layer.
 */
class LayerInitialization
{
 public:
  /**
   * Placeholder constructor, which does nothing. The actual initialization is
   * network_init.hpp/Initialize routine.
   */
  LayerInitialization()
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Layer initialization method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename MatType>
  void Initialize(MatType& /* W */, const size_t /* rows */, const size_t /* cols */)
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Layer initialization method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename MatType>
  void Initialize(MatType& /* W */,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * Layer initialization method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slices Number of slices.
   */
  template<typename CubeType>
  void Initialize(CubeType& /* W */,
                  const size_t /* rows */,
                  const size_t /* cols */,
                  const size_t /* slices */)
  {
     // Nothing to do here.
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * Layer initialization method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename CubeType>
  void Initialize(CubeType& /* W */,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    // Nothing to do here.
  }
 
  /**
   * Serialize the initialization.  (Nothing to serialize for this one.)
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
}; // class LayerInitialization

//! Initialization traits of the Layer initialization rule.
template<>
class InitTraits<LayerInitialization>
{
 public:
  //! The Layer initialization rule is applied over the entire network.
  static const bool UseLayer = false;
  //! The layer initialization rule uses the network to set
  //! the weights.
  static const bool UseNetwork = true;
};

} // namespace mlpack

#endif
