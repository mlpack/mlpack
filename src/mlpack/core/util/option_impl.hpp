/**
 * @file option_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of template functions for the Option class.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_UTIL_OPTION_IMPL_HPP
#define __MLPACK_CORE_UTIL_OPTION_IMPL_HPP

// Just in case it has not been included.
#include "option.hpp"

namespace mlpack {
namespace util {

/**
 * Registers a parameter with CLI.
 */
template<typename N>
Option<N>::Option(bool ignoreTemplate,
                  N defaultValue,
                  const std::string& identifier,
                  const std::string& description,
                  const std::string& alias,
                  bool required)
{
  if (ignoreTemplate)
  {
    CLI::Add(identifier, description, alias, required);
  }
  else
  {
    CLI::Add<N>(identifier, description, alias, required);
    CLI::GetParam<N>(identifier) = defaultValue;
  }
}


/**
 * Registers a flag parameter with CLI.
 */
template<typename N>
Option<N>::Option(const std::string& identifier,
                  const std::string& description,
                  const std::string& alias)
{
  CLI::AddFlag(identifier, description, alias);
}

}; // namespace util
}; // namespace mlpack

#endif
