/**
 * @file methods/softmax_regression/softmax_regression_utils.hpp
 * @author Dirk Eddelbuettel
 *
 * Prediction step used in 'classify' and 'probabilities'.
 * Used only for bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_UTILS_HPP
#define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_UTILS_HPP

#include "softmax_regression_utils_impl.hpp"

namespace mlpack {

// Test the accuracy of the model.
template<typename Model>
inline void RunPredictionStep(util::Params& params,
                              util::Timers& timers,
                              const size_t numClasses,
                              const Model& model,
                              const bool retPreds,
                              const bool retProbas);

} // namespace mlpack

#endif
