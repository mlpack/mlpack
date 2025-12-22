/**
 * @file methods/tsne_functions/tsne_function.hpp
 * @author Ranjodh Singh
 *
 * Maps each tsne method (ExactTSNE, BarnesHutTSNE, DualTreeTSNE) to its
 * corresponding objective function using type traits and a convenience alias.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_FUNCTION_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_FUNCTION_HPP

#include "../tsne_methods.hpp"
#include "tsne_exact_function.hpp"
#include "tsne_approx_function.hpp"

namespace mlpack {

template <typename MatType, typename DistanceType, typename TSNEMethod>
class TSNEFunctionTraits
{
  using type = TSNEBarnesHutFunction<MatType, DistanceType>;
};

template <typename MatType, typename DistanceType>
class TSNEFunctionTraits<MatType, DistanceType, ExactTSNE>
{
 public:
  using type = TSNEExactFunction<MatType, DistanceType>;
};

template <typename MatType, typename DistanceType>
class TSNEFunctionTraits<MatType, DistanceType, DualTreeTSNE>
{
 public:
  using type = TSNEDualTreeFunction<MatType, DistanceType>;
};

template <typename MatType, typename DistanceType>
class TSNEFunctionTraits<MatType, DistanceType, BarnesHutTSNE>
{
 public:
  using type = TSNEBarnesHutFunction<MatType, DistanceType>;
};

// Convenience alias:
template <typename MatType, typename DistanceType, typename TSNEMethod>
using TSNEFunction = typename TSNEFunctionTraits<MatType,
                                                 DistanceType,
                                                 TSNEMethod>::type;

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_FUNCTION_HPP
