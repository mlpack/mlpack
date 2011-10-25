#ifndef MLPACK_CLI_OPTCLIN_IMPL_H
#define MLPACK_CLI_OPTCLIN_IMPL_H

#include "cli.hpp"
#include "log.h"

namespace mlpack {

/*
 * @brief Registers a parameter with CLI.
 *    This allows the registration of parameters at program start.
 */
template<typename N>
Option<N>::Option(bool ignoreTemplate,
                N defaultValue,
                const char* identifier,
                const char* description,
                const char* parent,
                bool required) {
  if (ignoreTemplate)
    CLI::Add(identifier, description, parent, required);
  else {
    CLI::Add<N>(identifier, description, parent, required);

    // Create the full pathname to set the default value.
    std::string pathname = CLI::SanitizeString(parent) + std::string(identifier);
    CLI::GetParam<N>(pathname.c_str()) = defaultValue;
  }
}


/*
 * @brief Registers a flag parameter with CLI.
 */
template<typename N>
Option<N>::Option(const char* identifier,
                  const char* description,
                  const char* parent) {
  CLI::AddFlag(identifier, description, parent);
}

}; // namespace mlpack

#endif
