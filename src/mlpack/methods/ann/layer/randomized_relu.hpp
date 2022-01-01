/**
 * @file methods/ann/layer/randomized_relu.hpp
 * @author Shubham Agrawal
 *
 * @code
 * @article{Bing Xu2015,
 *  author = {Bing Xu, Naiyan Wang, Tianqi Chen, Mu Li},
 *  title = {Empirical Evaluation of Rectified Activations in Convolutional 
 *      Network},
 *  year = {2015},
 *  url = {https://arxiv.org/pdf/1505.00853.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RRELU_HPP
#define MLPACK_METHODS_ANN_LAYER_RRELU_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The RReLU activation function (alpha is randomized), defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \max(x, alpha*x) \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class RReLU
{
 public:
  /**
   * Create the RReLU object using the specified parameters.
   * The non zero gradient will be adjusted by specifying the parameter
   * lowerbound, and higherbound from range 1 to inf.
   * Default (lowerbound = 3.0, higherbound = 8.0)
   *
   * @param lowerbound Lowerbound of uniform distribution
   * @param higherbound Higherbound of uniform distribution
   */
  RReLU(const double lowerbound = 3.0,
        const double higherbound = 8.0);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& input, const DataType& gy, DataType& g);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the non zero gradient.
  double const& Alpha() const { return alpha; }
  //! Modify the non zero gradient.
  double& Alpha() { return alpha; }

	 //! The value of the deterministic parameter.
  bool const& Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

	//! Get the lowerbound parameter.
  double const& LowerBound() const { return lowerbound; }
  //! Modify the lowerbound parameter.
  void LowerBound(const double lo)
	{
		if (lo < 1.0)
			Log::Fatal << "Lowerbound must be greater than 1.0" << std::endl;
		lowerbound = lo;
	}

	//! Get the higherbound parameter.
  double const& HigherBound() const { return higherbound; }
  //! Modify the upperbound parameter.
  void HigherBound(const double hi)
	{
		if (hi < 1.0)
			Log::Fatal << "Higherbound must be greater than 1.0" << std::endl;
		higherbound = hi;
	}

  //! Get size of weights.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

	//! Lower bound of uniform distribution
	double lowerbound;

	//! Higher bound of uniform distribution
	double higherbound;

  //! If true randomization of alpha is disabled, see notes above.
  bool deterministic;

  //! Locally-stored Leakyness Parameter in the range 0 <alpha< 1
  double alpha;
}; // class RReLU

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "randomized_relu_impl.hpp"

#endif
