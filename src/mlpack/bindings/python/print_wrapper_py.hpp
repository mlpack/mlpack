/**
 * @file print_wrapper_py.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_WRAPPER_PY_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_WRAPPER_PY_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace python {

void PrintWrapperPY(const std::string& category,
                    const std::string& groupName,
                    const std::string& validMethods);

} // namespace mlpack
} // namespace bindings
} // namespace python

#endif
