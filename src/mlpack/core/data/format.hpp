/**
 * @file core/data/format.hpp
 * @author Ryan Curtin
 *
 * Define the formats that can be used by mlpack's Load() and Save() mechanisms
 * via cereal.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_FORMATS_HPP
#define MLPACK_CORE_DATA_FORMATS_HPP

namespace mlpack {
namespace data {

//! Define the formats we can read through cereal.
enum format
{
  autodetect,
  json,
  xml,
  binary
};

} // namespace data
} // namespace mlpack

#endif
