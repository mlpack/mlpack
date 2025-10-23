/**
 * @file methods/tsne_functions/tsne_function.hpp
 * @author Ranjodh Singh
 *
 * Maps each tsne methods (ExactTSNE, BarnesHutTSNE, DualTreeTSNE) to their
 * corresponding objective functions using type traits and a convenience alias.
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

template <typename TSNEMethod>
class TSNEFunctionTraits
{
  using type = TSNEApproxFunction<false>;
};

template <>
class TSNEFunctionTraits<ExactTSNE>
{
 public:
  using type = TSNEExactFunction<>;
};

template <>
class TSNEFunctionTraits<DualTreeTSNE>
{
 public:
  using type = TSNEApproxFunction<true>;
};

template <>
class TSNEFunctionTraits<BarnesHutTSNE>
{
 public:
  using type = TSNEApproxFunction<false>;
};

// Convenience alias:
template <typename TSNEMethod>
using TSNEFunction = typename TSNEFunctionTraits<TSNEMethod>::type;

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_FUNCTION_HPP
