/**
 * @file one_hot_layer.hpp
 * @author Shangtong Zhang
 *
 * Definition of the OneHotLayer class, which implements a standard network
 * layer.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_EMPTY_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_EMPTY_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * A layer do not do anything
 */
class EmptyLayer
{
 public:
  /**
   * Create the OneHotLayer object.
   */
  EmptyLayer()
  {
    // Nothing to do here.
  }

  template<typename DataType>
  void CalculateError(const DataType&,
                      const DataType&,
                      DataType&)
  {    
  }

  /*
   * Calculate the output class using the specified input activation.
   *
   * @param inputActivations Input data used to calculate the output class.
   * @param output Output class of the input activation.
   */
  template<typename DataType>
  void OutputClass(const DataType& inputActivations, DataType& output)
  {
    output = inputActivations;
    output.zeros();

    arma::uword maxIndex;
    inputActivations.max(maxIndex);
    output(maxIndex) = 1;
  }

  template<typename Archive>
  void Serialize(Archive&, const unsigned int /* version */)
  {
  }
}; // class OneHotLayer

//! Layer traits for the one-hot class classification layer.
template <>
class LayerTraits<EmptyLayer>
{
 public:
  static const bool IsBinary = true;
  static const bool IsOutputLayer = true;
  static const bool IsBiasLayer = false;
  static const bool IsConnection = false;
};

}; // namespace ann
}; // namespace mlpack


#endif
