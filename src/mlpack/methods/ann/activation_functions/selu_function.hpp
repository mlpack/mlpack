/**
 * @file selu_function.hpp
 * @author Dakshit Agrawal
 *
 * Definition and implementation of the SELU function as introduced by
 * Klambauer et. al. in Self Neural Networks.  The SELU activation
 * function keeps the mean and variance of the input invariant.
 *
 * For more information, see the following paper.
 *
 * @code
 * @article{Klambauer2017,
 *   author  = {Gunter Klambauer and Thomas Unterthiner and
 *              Andreas Mayr},
 *   title   = {Self-Normalizing Neural Networks},
 *   journal = {Advances in Neural Information Processing Systems},
 *   year    = {2017}
 * }
 * }
 * @endcode
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SELU_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_SELU_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>

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
 *     lambda * (y + alpha) & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 *
 *
 *
 * NOTE:
 * Make sure to use this activation function with normalized inputs and
 * weights initialized with Lecun Normal Initialization.
 */

class SELUFunction
{
 public:
    /**
     * Computes the SELU activation function.
     *
     * @param x Input data.
     * @return f(x).
     */
    static double Fn(const double x)
    {
        if (x < DBL_MAX) {
            return (x > 0) ? lambda * x : lambda * alpha * (std::exp(x) - 1);
        }
        return 1.0;
    }

    /**
     * Computes the SELU activation function using a dense matrix as input.
     *
     * @param x Input data.
     * @param y The resulting output activation.
     */
    template<typename eT>
    static void Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
    {
        y = x;

        for (size_t i = 0; i < x.n_elem; i++)
        {
            y(i) = Fn(x(i));
        }
    }

    /**
     * Computes the SELU activation function using a 3rd-order
     * tensor as input.
     *
     * @param x Input data.
     * @param y The resulting output activation.
     */
    template<typename eT>
    static void Fn(const arma::Cube<eT>& x, arma::Cube<eT>& y)
    {
        y = x;
        for (size_t s = 0; s < x.n_slices; s++)
        {
            Fn(x.slice(s), y.slice(s));
        }
    }

    /**
     * Computes the first derivative of the SELU activation function.
     *
     * @param x Input data.
     * @return f'(x)
     */
    static double Deriv(const double y)
    {
        return (y > 0) ? lambda : lambda * (y + alpha);
    }

    /**
     * Computes the first derivatives of the SELU activation function
     * using a dense matrix as input.
     *
     * @param y Input activations.
     * @param x The resulting derivatives.
     */
    template<typename InputType, typename OutputType>
    static void Deriv(const InputType& y, OutputType& x)
    {
        x = y;

        for (size_t i = 0; i < y.n_elem; i++)
        {
            x(i) = Deriv(y(i));
        }
    }

    //! Get the non zero alpha.
    static double const& Alpha() { return alpha; }


    //! Get the non zero lambda.
    static double const& Lambda() { return lambda; }


 private:
    // The following default constant values are for input whose mean is 0 and
    // variance is 1, i.e. input is normalized.

    constexpr static double alpha = 1.6732632423543774;

    constexpr static double lambda = 1.0507009873554802;
}; // class SELUFunction

} // namespace ann
} // namespace mlpack

#endif

