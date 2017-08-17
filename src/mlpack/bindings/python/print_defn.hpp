/**
 * @file print_defn.hpp
 * @author Ryan Curtin
 *
 * Print the definition of a Python parameter.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_DEFN_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_DEFN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Print the definition for a Python binding parameter to stdout.  This is the
 * definition in the function declaration.
 */
template<typename T>
void PrintDefn(const util::ParamData& d,
               const void* /* input */,
               void* /* output */)
{
  // Make sure that we don't use names that are Python keywords.
  std::string name = (d.name == "lambda") ? "lambda_" : d.name;

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
