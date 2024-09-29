/**
 * @file core/util/forward.hpp
 * @author Ryan Curtin
 *
 * Forward declaration of components from other subdirectories necessary for
 * various util implementations.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_FORWARD_HPP
#define MLPACK_CORE_UTIL_FORWARD_HPP

#include <mlpack/base.hpp>

// Required forward declarations.
namespace mlpack {

class IO;

namespace util {

class Timers;

} // namespace util
}

#include "params.hpp"

namespace mlpack {
namespace data {

class IncrementPolicy;

template<typename PolicyType, typename InputType>
class DatasetMapper;

using DatasetInfo = DatasetMapper<IncrementPolicy, std::string>;

// This is a forward declaration of a function that just calls std::get(); but,
// we cannot use std::get directly because we have only forward-declared
// DatasetInfo.
void CheckCategoricalParam(util::Params& p, const std::string& paramName);

} // namespace data
} // namespace mlpack

#endif
