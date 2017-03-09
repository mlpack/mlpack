/**
 * @file batchnorm.hpp
 * @author Marcus Edel
 *
 * Definition of the Batch Normalisation layer class as proposed by Ioffe et.al
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BATCHNORM_HPP
#define MLPACK_METHODS_ANN_LAYER_BATCHNORM_HPP

#include <mlpack/prereqs.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class BatchNorm
{
 public:

  BatchNorm();

  BatchNorm(const size_t size, const double eps);

  void Reset();

  template<typename eT>
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);

  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  template<typename eT>
  void Gradient(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  OutputDataType& Mean() { return mean; }

  OutputDataType& Variance() { return variance; }

  OutputDataType& Gamma() { return gamma; }

  OutputDataType& Beta() { return beta; }



  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:

  OutputDataType gamma;

  OutputDataType beta;

  OutputDataType weights;

  double eps;

  bool deterministic;

  size_t size;

  OutputDataType mean;

  // OutputDataType trainingMean;

  OutputDataType variance;

  // OutputDataType trainingVariance;

  arma::running_stat_vec<arma::colvec> stats;

  OutputDataType gradient;

  OutputDataType delta;

   //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
};

}
}

#include "batchnorm_impl.hpp"

#endif
