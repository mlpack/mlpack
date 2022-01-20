/**
 * @file bindings/python/print_defn.hpp
 * @author Ryan Curtin
 *
 * Print the definition of a Python parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_DEFN_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_DEFN_HPP

#include <mlpack/prereqs.hpp>
#include "wrapper_functions.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Print the definition for a Python binding parameter to stdout.  This is the
 * definition in the function declaration.
 */
template<typename T>
void PrintDefn(util::ParamData& d,
               const void* /* input */,
               void* /* output */)
{
  // Make sure that we don't use names that are Python keywords.
  std::string name = GetValidName(d.name);

  std::cout << name;
  if (std::is_same<T, bool>::value)
    std::cout << "=False";
  else if (!d.required)
    std::cout << "=None";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
