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
                const char* parent,
                bool required)
{
  if (ignoreTemplate)
  {
    CLI::Add(identifier, description, parent, required);
  }
  else
  {
    CLI::Add<N>(identifier, description, parent, required);

    CLI::GetParam<N>(identifier) = defaultValue;
  }
}


/**
 * Registers a flag parameter with CLI.
 */
template<typename N>
Option<N>::Option(const char* identifier,
                  const char* description,
                  const char* parent)
{
  CLI::AddFlag(identifier, description, parent);
}

}; // namespace io
}; // namespace mlpack

#endif
