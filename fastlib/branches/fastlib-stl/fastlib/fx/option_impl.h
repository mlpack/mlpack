#ifndef MLPACK_IO_OPTION_IMPL_H
#define MLPACK_IO_OPTION_IMPL_H

#include "io.h"

namespace mlpack {

/*
 * @brief Registers a parameter with IO.  
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
    IO::Add(identifier, description, parent, required);
  else {
    IO::Add<N>(identifier, description, parent, required);

    //Create the full pathname.
    std::string pathname = IO::SanitizeString(parent) + std::string(identifier);
    IO::GetParam<N>(pathname.c_str()) = defaultValue;
  }
}


/*
 * @brief Registers a flag parameter with IO.
 */
template<typename N>
Option<N>::Option(const char* identifier,
                  const char* description,
                  const char* parent) {
  IO::AddFlag(identifier, description, parent);
}

}; // namespace mlpack

#endif 
