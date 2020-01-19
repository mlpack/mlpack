/**
 * @file mish_function.hpp
 * @author Kartik Dutt
 *
 * Definition and implementation of the Mish function as described by
 * Diganta Misra.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{
 *   author = {Diganta Misra},
 *   title = {Mish: Self Regularized Non-Monotonic Neural Activation Function},
 *   year = {2019}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MISH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_MISH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <algorithm>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The mish function, defined by
 *
 * @f{eqnarray*}{
 * f(x) = x * tanh(ln(1+e^x))
 * f'(x) = tanh(ln(1+e^x)) + x * ((1 - tanh^2(ln(1+e^x))) * frac{1}{1 + e^{-x}})
 * }
 */
class MishFunction
{
 public:

    /**
     * Computes the Mish function.
     *
     * @param x Input data.
     * @return f(x).
     */
    static double Fn(const double x)
    {
        return x * std::tanh(SoftplusFunction::Fn(x));
    }

    /**
     * Computes the mish function.
     *
     * @param x Input data.
     * @param y The resulting output activation.
     */
    template <typename InputVecType, typename OutputVecType>
    static void Fn(const InputVecType &x, OutputVecType &y)
    {
        y.set_size(arma::size(x));

        for (size_t i = 0; i < x.n_elem; i++)
            y(i) = Fn(x(i));
    }

    /**
     * Computes the first derivative of the swish function.
     *
     * @param y Input data.
     * @return f'(x)
     */
    static double Deriv(const double y)
    {
        return std::tanh(SoftplusFunction::Fn(y)) +
            y * (SoftplusFunction::Deriv(y) *
            (1 - std::pow(SoftplusFunction::Fn(y), 2)));
    }

    /**
     * Computes the first derivatives of the mish function.
     * 
     * @param y Input activations.
     * @param x The resulting derivatives.
     */
    template <typename InputVecType, typename OutputVecType>
    static void Deriv(const InputVecType &y, OutputVecType &x)
    {
        InputVecType softPlusY;
        InputVecType derivSoftPlusY;
        SoftplusFunction::Fn(y, softPlusY);
        SoftplusFunction::Deriv(y,derivSoftPlusY);
        x = arma::tanh(softPlusY) +
            y * ((1 - arma::pow(softPlusY, 2)) * derivSoftPlusY);
    }
}; // class MishFunction

} // namespace ann
} // namespace mlpack

#endif
