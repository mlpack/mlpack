/**
 * @file core/data/extension.hpp
 * @author Ryan Curtin
 *
 * Given a filename, extract its extension.  This is used by data::Load() and
 * data::Save().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_EXTENSION_HPP
#define MLPACK_CORE_DATA_EXTENSION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

inline std::string Extension(const std::string& filename)
{
  const size_t ext = filename.rfind('.');
  std::string extension;
  if (ext == std::string::npos)
    return extension;

  extension = filename.substr(ext + 1);
  std::transform(extension.begin(), extension.end(), extension.begin(),
      ::tolower);

  return extension;
}

} // namespace data
} // namespace mlpack

#endif
