/**
 * @file range_serialization.hpp
 * @author Ryan Curtin
 *
 * Implementations of Serialize() for math::Range.
 */
#ifndef MLPACK_CORE_MATH_RANGE_SERIALIZATION_HPP
#define MLPACK_CORE_MATH_RANGE_SERIALIZATION_HPP

#include "range.hpp"
#include <mlpack/core/data/serialization.hpp>

namespace mlpack {
namespace math {

//! Serialize the range.
template<typename T>
template<typename Archive>
void RangeType<T>::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(hi, "hi");
  ar & data::CreateNVP(lo, "lo");
}

} // namespace math
} // namespace mlpack

#endif
