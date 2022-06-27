/**
 * @file core/util/forward.hpp
 * @author Ryan Curtin
 *
 * Forward declaration of components from other subdirectories necessary for
 * various util implementations.
 */
#ifndef MLPACK_CORE_UTIL_FORWARD_HPP
#define MLPACK_CORE_UTIL_FORWARD_HPP

#include <mlpack/base.hpp>

// Required forward declarations.
namespace mlpack {

class IO;

namespace util {

class Params;
class Timers;

} // namespace util

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
