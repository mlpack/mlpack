/**
 * @file formats.hpp
 * @author Ryan Curtin
 *
 * Define the formats that can be used by mlpack's Load() and Save() mechanisms
 * via boost::serialization.
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
