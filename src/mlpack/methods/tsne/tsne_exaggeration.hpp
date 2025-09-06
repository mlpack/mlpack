/**
 * @file methods/tsne/tsne_exaggeration.hpp
 * @author Ranjodh Singh
 *
 * A callback used by the default TSNE optimizer for early exaggeration.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_EXAGGERATION_HPP
#define MLPACK_METHODS_TSNE_TSNE_EXAGGERATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack
{

class TSNEExaggeration
{
 public:
  TSNEExaggeration(size_t earlyIters) : earlyIters(earlyIters) {};

  template <typename OptimizerType, typename FunctionType, typename MatType>
  void BeginOptimization(OptimizerType& /* optimizer */,
                         FunctionType& function,
                         MatType& /* coordinates */)
  {
    function.StartExaggerating();
  }

  template <typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& function,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double /* objective */)
  {
    if (epoch == earlyIters)
    {
      function.StopExaggerating();
    }

    return false;
  }

 private:
  size_t earlyIters;
};

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_EXAGGERATION_HPP
