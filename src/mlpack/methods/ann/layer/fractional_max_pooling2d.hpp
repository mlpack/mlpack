/**
 * @file methods/ann/layer/fractional_max_pooling2d.hpp
 * @author Mayank Raj
 *
 * Definition of the FractionalMaxPooling2D class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_FRACTIONAL_MAX_POOLING2D_HPP
#define MLPACK_METHODS_ANN_LAYER_FRACTIONAL_MAX_POOLING2D_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

template <typename MatType = arma::mat>
class FractionalMaxPooling2DType : public Layer<MatType>
{
 public:
  //! Create the FractionalMaxPooling2D object with default parameters.
  FractionalMaxPooling2DType();

  /**
   * Create the FractionalMaxPooling2D object.
   *
   * @param poolingRatio The ratio by which the input dimensions will be 
   *                     divided to determine the output dimensions.
   */
  FractionalMaxPooling2DType(const double poolingRatio);

  //! Virtual destructor.
  virtual ~FractionalMaxPooling2DType() {}

  //! Copy the given FractionalMaxPooling2D.
  FractionalMaxPooling2DType(const FractionalMaxPooling2DType& other);

  //! Take ownership of the given FractionalMaxPooling2D.
  FractionalMaxPooling2DType(FractionalMaxPooling2DType&& other);

  //! Copy the given FractionalMaxPooling2D.
  FractionalMaxPooling2DType& operator=(const FractionalMaxPooling2DType& other);

  //! Take ownership of the given FractionalMaxPooling2D.
  FractionalMaxPooling2DType& operator=(FractionalMaxPooling2DType&& other);

    //! Clone the AdaptiveMaxPoolingType object.
    //! This handles polymorphism correctly.
    FractionalMaxPooling2DType* Clone() const
    {
      return new FractionalMaxPooling2DType(*this);
    }

  void Forward(const MatType& input, MatType& output);

  void Backward(const MatType& input,
                const MatType& gy,
                MatType& g);

  double const& PoolingRatio() const { return poolingRatio; }

  double& PoolingRatio() { return poolingRatio; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! The ratio by which the input dimensions will be divided.
  double poolingRatio;
};

// Convenience typedefs.
typedef FractionalMaxPooling2DType<arma::mat> FractionalMaxPool2D;

} // namespace mlpack

// Include implementation.
#include "fractional_max_pooling2d_impl.hpp"

#endif