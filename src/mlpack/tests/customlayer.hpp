#ifndef CUSTOMLAYER_HPP
#define CUSTOMLAYER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>


namespace mlpack {
namespace ann{
  /**
   * Standard hyperbolic tangent layer.
   */
  template <
      class ActivationFunction = LogisticFunction,
      typename InputDataType = arma::mat,
      typename OutputDataType = arma::mat
  >
  using CustomLayer = BaseLayer<
      ActivationFunction, InputDataType, OutputDataType>;

} // namespace ann
} // namespace mlpack

#endif // CUSTOMLAYER_HPP
