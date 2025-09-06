/**
 * @file methods/tsne_functions/tsne_function.hpp
 * @author Ranjodh Singh
 *
 * Convenience include for tsne functions.
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

namespace mlpack
{

template <typename TSNEPolicy>
class TSNEFunction
{
  /* Nothing To Do Here */
};

template <>
class TSNEFunction<ExactTSNE>
{
 public:
  using type = TSNEExactFunction<>;
};

template <>
class TSNEFunction<DualTreeTSNE>
{
 public:
  using type = TSNEApproxFunction<true>;
};

template <>
class TSNEFunction<BarnesHutTSNE>
{
 public:
  using type = TSNEApproxFunction<false>;
};

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_FUNCTION_HPP
