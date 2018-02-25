/**
 * @file selu.hpp
 * @author Dakshit Agrawal
 *
 * Definition of the SELU activation function as descibed by Gunter Klambauer,
 * Thomas Unterthiner and Andreas Mayr.
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{Clevert2015,
 *   author  = {Gunter Klambauer and Thomas Unterthiner and
 *              Andreas Mayr},
 *   title   = {Self-Normalizing Neural Networks},
 *   journal = {Advances in Neural Information Processing Systems},
 *   year    = {2017}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SELU_HPP
#define MLPACK_METHODS_ANN_LAYER_SELU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
    namespace ann /** Artificial Neural Network. */ {

    /**
    * The SELU activation function, defined by
    *
    * @f{eqnarray*}{
    * f(x) &=& \left\{
    *   \begin{array}{lr}
    *    lambda * x & : x > 0 \\
    *    lambda * alpha(e^x - 1) & : x \le 0
    *   \end{array}
    * \right. \\
    * f'(x) &=& \left\{
    *   \begin{array}{lr}
    *     lambda & : x > 0 \\
    *     lambda(y + alpha) & : x \le 0
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
        class SELU
        {
        public:

            /**
             * Create the SELU object using the specified parameters.  Default values of alpha and lambda are set for
             * normalized inputs, i.e. inputs having mean = 0 and variance = 1.
             *
             * @param alpha The scale parameter for negative factor (Default alpha = 1.6732632423543772848170429916717)
             * @param lambda The new parameter introduced for scaling ELU activation function (Default lambda = 1.0507009873554804934193349852946)
             * @param inSize The number of input units.
             * @param outSize The number of output units.
             */

            SELU(const double alpha = 1.6732632423543772848170429916717,
                 const  double lambda = 1.0507009873554804934193349852946,
                const size_t inSize = 1,
                const size_t outSize
            );

            /**
             * Ordinary feed forward pass of a neural network, evaluating the function
             * f(x) by propagating the activity forward through f.
             *
             * @param input Input data used for evaluating the specified function.
             * @param output Resulting output activation.
             */
            template<typename InputType, typename OutputType>
            void Forward(const InputType&& input, OutputType&& output);

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
            void Backward(const DataType&& input, DataType&& gy, DataType&& g);

            /**
             * Calculate the gradient using the output delta and the input activation.
             *
             * @param input The propagated input.
             * @param error The calculated error.
             * @param gradient The calculated gradient.
             */


            template<typename DataType>
            void Gradient(DataType&& input,
                          DataType&& error,
                          DataType&& gradient);

            //! Get the input parameter.
            InputDataType const& InputParameter() const { return inputParameter; }
            //! Modify the input parameter.
            InputDataType& InputParameter() { return inputParameter; }

            //! Get the weights.
            OutputDataType const& Weights() const { return weights; }
            //! Modify the weights.
            OutputDataType& Weights() { return weights; }

            //! Get the biases.
            OutputDataType const& Bias() const { return bias; }
            //! Modify the biases.
            OutputDataType& Bias() { return bias; }

            //! Get the output parameter.
            OutputDataType const& OutputParameter() const { return outputParameter; }
            //! Modify the output parameter.
            OutputDataType& OutputParameter() { return outputParameter; }

            //! Get the delta.
            OutputDataType const& Delta() const { return delta; }
            //! Modify the delta.
            OutputDataType& Delta() { return delta; }

            //! Get the insize.
            size_t const& Insize() const { return inSize; }
            //! Modify the insize.
            size_t & Insize() { return inSize; }

            //! Get the outsize.
            size_t const& Outsize() const { return outSize; }
            //! Modify the outsize.
            size_t & Outsize() { return outSize; }

            //! Get the non zero gradient.
            double const& Alpha() const { return alpha; }

            //! Get the non zero scale
            double const& Lambda() const { return lambda; }

            /**
             * Serialize the layer.
             */
            template<typename Archive>
            void serialize(Archive& ar, const unsigned int /* version */);

        private:
            /**
             * Computes the SELU function
             *
             * @param x Input data.
             * @return f(x).
             */
            double Fn(const double x)
            {
                if (x < DBL_MAX) {
                    return (x > 0) ? lambda * x : lambda * (alpha * (std::exp(x) - 1));
                }
                return 1.0;
            }

            /**
             * Computes the SELU function using a dense matrix as input.
             *
             * @param x Input data.
             * @param y The resulting output activation.
             */

            template<typename InputType, typename OutputType>
            void Fn(const InputType& x, OutputType& y)
            {
                y = x;

                for (size_t i = 0; i < x.n_elem; i++)
                {
                    y(i) = Fn(x(i));
                }
            }

            /**
             * Computes the first derivative of the SELU function.
             *
             * @param x Input data.
             * @return f'(x)
             */
            double Deriv(const double y)
            {
                return (y > 0) ? lambda : lambda * (y + alpha);
            }

            /**
             * Computes the first derivative of the SELU function.
             *
             * @param y Input activations.
             * @param x The resulting derivatives.
             */

            template<typename InputType, typename OutputType>
            void Deriv(const InputType& x, OutputType& y)
            {
                y = x;

                for (size_t i = 0; i < x.n_elem; i++)
                {
                    y(i) = Deriv(x(i));
                }
            }

            //! Locally-stored delta object.
            OutputDataType delta;

            //! Locally-stored input parameter object.
            InputDataType inputParameter;

            //! Locally-stored output parameter object.
            OutputDataType outputParameter;

            //! Weights
            OutputDataType weights;

            //! Bias
            OutputDataType bias;

            //! Locally-stored number of input units.
            size_t inSize;

            //! Locally-stored number of output units.
            size_t outSize;

            //! SELU parameter alpha
            const double alpha;

            //! SELU parameter lambda
            const double lambda;
        }; // class SELU

    } // namespace ann
} // namespace mlpack

// Include implementation.
#include "selu_impl.hpp"

#endif //MLPACK_SELU_HPP
