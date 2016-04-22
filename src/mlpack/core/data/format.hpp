/**
 * @file formats.hpp
 * @author Ryan Curtin
 *
 * Define the formats that can be used by mlpack's Load() and Save() mechanisms
 * via boost::serialization.
 */
#ifndef MLPACK_CORE_DATA_FORMATS_HPP
#define MLPACK_CORE_DATA_FORMATS_HPP

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
