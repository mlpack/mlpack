/**
 * @file ada_delta_impl.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
 *
 * Implementation of the AdaDelta optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_DELTA_ADA_DELTA_IMPL_HPP
#define ENSMALLEN_ADA_DELTA_ADA_DELTA_IMPL_HPP

// In case it hasn't been included yet.
#include "ada_delta.hpp"

namespace ens {

inline AdaDelta::AdaDelta(const double stepSize,
                          const size_t batchSize,
                          const double rho,
                          const double epsilon,
                          const size_t maxIterations,
                          const double tolerance,
                          const bool shuffle) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              AdaDeltaUpdate(rho, epsilon))
{ /* Nothing to do. */ }

} // namespace ens

#endif
