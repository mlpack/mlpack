/**
 * @file extension.hpp
 * @author Ryan Curtin
 *
 * Given a filename, extract its extension.  This is used by data::Load() and
 * data::Save().
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
