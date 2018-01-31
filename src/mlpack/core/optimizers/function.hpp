/**
 * @file function.hpp
 * @author Ryan Curtin
 *
 * The Function class is a wrapper class for any objective function that
 * provides any of the functions that an optimizer might use.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include "add_evaluate_with_gradient.hpp"

namespace mlpack {
namespace optimization {

template<typename FunctionType>
class Function :
    public AddEvaluateWithGradient<FunctionType>
{ };

} // namespace optimization
} // namespace mlpack

// Include implementation.
//#include "function_impl.hpp"

#endif
