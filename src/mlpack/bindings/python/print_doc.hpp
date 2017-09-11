/**
 * @file print_doc.hpp
 * @author Ryan Curtin
 *
 * Print documentation (as part of a docstring) for a Python binding parameter.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_DOC_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_DOC_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>
#include "get_python_type.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Print the docstring documentation for a given parameter.  You are responsible
 * for setting up the line---this does not handle indentation or anything.  This
 * is meant to produce a line of documentation describing a single parameter.
 *
 * The indent parameter (void* input, which should be a pointer to a size_t)
 * should be passed to know how much to indent for a new line.
 *
 * @param d Parameter data struct.
 * @param input Pointer to size_t containing indent.
 * @param output Unused parameter.
 */
template<typename T>
void PrintDoc(const util::ParamData& d,
              const void* input,
              void* /* output */)
{
  const size_t indent = *((size_t*) input);
  std::ostringstream oss;
  oss << " - " << d.name << " (" << GetPythonType<T>(d) << "): " << d.desc;
  std::cout << util::HyphenateString(oss.str(), indent + 4);
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
