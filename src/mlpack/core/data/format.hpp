/**
 * @file formats.hpp
 * @author Ryan Curtin
 *
 * Define the formats that can be used by mlpack's Load() and Save() mechanisms
 * via boost::serialization.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_DATA_FORMATS_HPP
#define __MLPACK_CORE_DATA_FORMATS_HPP

namespace mlpack {
namespace data {

//! Define the formats we can read through boost::serialization.
enum format
{
  autodetect,
  text,
  xml,
  binary
};

} // namespace data
} // namespace mlpack

#endif
