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
