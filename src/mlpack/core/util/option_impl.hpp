/**
 * @file option_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of template functions for the Option class.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_OPTION_IMPL_HPP
#define MLPACK_CORE_UTIL_OPTION_IMPL_HPP

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

} // namespace util
} // namespace mlpack

#endif
