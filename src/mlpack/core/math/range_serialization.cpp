/**
 * @file range_serialization.cpp
 * @author Ryan Curtin
 *
 * Instantiation of Serialize() operators for math::Range.
 */
#include <limits>
#include "range_serialization.hpp"

using namespace mlpack;
using namespace mlpack::math;

MLPACK_SERIALIZATION_INSTANTIATE(Range);
