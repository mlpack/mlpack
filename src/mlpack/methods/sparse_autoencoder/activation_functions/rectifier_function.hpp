/**
 * @file rectifier_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the rectifier function as described by
 * V. Nair and G. E. Hinton.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{NairHinton2010,
 *   author = {Vinod Nair, Geoffrey E. Hinton},
 *   title = {Rectified Linear Units Improve Restricted Boltzmann Machines},
 *   year = {2010}
 * }
 * @endcode
 */
#ifndef __MLPACK_METHODS_NN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP
#define __MLPACK_METHODS_NN_ACTIVATION_FUNCTIONS_RECTIFIER_FUNCTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

namespace mlpack {
namespace nn /** Neural Network. */ {
	
using RectifierFunction = ann::RectifierFunction;

}; // namespace nn
}; // namespace mlpack

#endif
