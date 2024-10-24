/**
 * @file radial_basis_function.hpp
 * @author Himanshu Pathak
 *
 * Definition of the Radial Basis Function module class.
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RADIAL_BASIS_FUNCTION_HPP
#define MLPACK_METHODS_ANN_LAYER_RADIAL_BASIS_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/activation_functions/gaussian_function.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Radial Basis Function layer. The RBFType class, when
 * used with a non-linear activation function, acts as a Radial Basis Function
 * which can be used with a feed-forward neural network.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{Volume 51: Artificial Intelligence and Statistics,
 *   author  = {Qichao Que, Mikhail Belkin},
 *   title   = {Back to the Future: Radial Basis Function Networks Revisited},
 *   year    = {2016},
 *   url     = {http://proceedings.mlr.press/v51/que16.pdf},
 * }
 * @endcode
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 * @tparam Activation Type of the activation function (GaussianFunction).
 */

template <
    typename MatType = arma::mat,
    typename Activation = GaussianFunction
>
class RBFType : public Layer<MatType>
{
 public:
  //! Create the RBFType object.
  RBFType();

  /**
   * Create the Radial Basis Function layer object using the specified
   * parameters.
   *
   * @param outSize The number of output units.
   * @param centres The centres calculated using k-means of data.
   * @param betas The beta value to be used with centres.
   */
  RBFType(const size_t outSize,
          MatType& centres,
          double betas = 0);

  //! Clone the LinearType object. This handles polymorphism correctly.
  RBFType* Clone() const { return new RBFType(*this); }

  // Virtual destructor.
  virtual ~RBFType() { }

  //! Copy the given RBFType layer.
  RBFType(const RBFType& other);
  //! Take ownership of the given RBFType layer.
  RBFType(RBFType&& other);
  //! Copy the given RBFType layer.
  RBFType& operator=(const RBFType& other);
  //! Take ownership of the given RBFType layer.
  RBFType& operator=(RBFType&& other);

  /**
   * Ordinary feed forward pass of the radial basis function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of the radial basis function.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& /* gy */,
                MatType& /* g */);

  //! Compute the output dimensions of the layer given `InputDimensions()`.  The
  //! RBFType layer flattens the input.
  void ComputeOutputDimensions();

  //! Get the size of the weights.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored the betas values.
  double betas;

  //! Locally-stored the learnable centre of the shape.
  MatType centres;

  //! Locally-stored the output distances of the shape.
  MatType distances;
}; // class RBFType

using RBF = RBFType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "radial_basis_function_impl.hpp"

#endif
