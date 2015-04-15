/**
 * @file extension.hpp
 * @author Ryan Curtin
 *
 * Given a filename, extract its extension.  This is used by data::Load() and
 * data::Save().
 */
#ifndef __MLPACK_CORE_DATA_EXTENSION_HPP
#define __MLPACK_CORE_DATA_EXTENSION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

std::string Extension(const std::string& filename)
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
