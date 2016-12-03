/**
 * @file concat_performance.hpp
 * @author Marcus Edel
 *
 * Definition of the ConcatPerformance class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_PERFORMANCE_HPP

#include <mlpack/core.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include "layer_types.hpp"
#include "layer_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the concat performance class. The class works as a
 * feed-forward fully connected network container which plugs performance layers
 * together.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename OutputLayerType = NegativeLogLikelihood<>,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class ConcatPerformance
{
 public:
  /**
   * Create the ConcatPerformance object.
   *
   * @param inSize The number of inputs.
   * @param outputLayer Output layer used to evaluate the network.
   */
  ConcatPerformance(const size_t inSize,
                    OutputLayerType&& outputLayer = OutputLayerType()) :
      inSize(inSize),
      outputLayer(std::move(outputLayer))
  {
    /* Nothing to do here. */
  }

  /*
   * Computes the Negative log likelihood.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  double Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& target)
  {
    const size_t elements = input.n_elem / inSize;

    double output = 0;
    for (size_t i = 0; i < input.n_elem; i+= elements)
    {
      arma::mat subInput = input.submat(i, 0, i + elements - 1, 0);
      output += outputLayer.Forward(std::move(subInput), std::move(target));
    }

    return output;
  }

  /**
   * Ordinary feed backward pass of a neural network. The negative log
   * likelihood layer expectes that the input contains log-probabilities for
   * each class. The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param input The propagated input activation.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   * @param output The calculated error.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                const arma::Mat<eT>&& target,
                arma::Mat<eT>&& output)
  {
    const size_t elements = input.n_elem / inSize;

    arma::mat subInput = input.submat(0, 0, elements - 1, 0);
    arma::mat subOutput;

    outputLayer.Backward(std::move(subInput), std::move(target),
        std::move(subOutput));

    output = arma::zeros(subOutput.n_elem, inSize);
    output.col(0) = subOutput;

    for (size_t i = elements, j = 0; i < input.n_elem; i+= elements, j++)
    {
      subInput = input.submat(i, 0, i + elements - 1, 0);
      outputLayer.Backward(std::move(subInput), std::move(target),
        std::move(subOutput));

      output.col(j) = subOutput;
    }
  }

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

 private:
  //! Locally-stored number of inputs.
  size_t inSize;

  //! Instantiated outputlayer used to evaluate the network.
  OutputLayerType outputLayer;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class ConcatPerformance

}; // namespace ann
}; // namespace mlpack

#endif
