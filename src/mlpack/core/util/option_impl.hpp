/**
 * @file option_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of template functions for the Option class.
 */
#ifndef __MLPACK_CORE_IO_OPTION_IMPL_HPP
#define __MLPACK_CORE_IO_OPTION_IMPL_HPP

// Just in case it has not been included.
#include "option.hpp"

namespace mlpack {
namespace io {

/**
 * Registers a parameter with CLI.
 */
template<typename N>
Option<N>::Option(bool ignoreTemplate,
                N defaultValue,
                const char* identifier,
                const char* description,
                const char* alias,
                bool required)
{
  if (ignoreTemplate)
  {
    if (alias == NULL)
      alias = "";

    CLI::Add(identifier, description, alias, required);
  }
  else
  {
    if (alias == NULL)
      alias = "";

    CLI::Add<N>(identifier, description, alias, required);

    CLI::GetParam<N>(identifier) = defaultValue;
  }
}


/**
 * Registers a flag parameter with CLI.
 */
template<typename N>
Option<N>::Option(const char* identifier,
                  const char* description,
                  const char* alias)
{
  if (alias == NULL)
    alias = "";

  CLI::AddFlag(identifier, description, alias);
}

}; // namespace io
}; // namespace mlpack

#endif
