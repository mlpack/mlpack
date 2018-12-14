/**
 * @file get_printable_type.hpp
 * @author Ryan Curtin
 *
 * Get the printable type of the parameter.  This depends on
 * BindingInfo::Language() to choose which language to return the type for.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_TYPE_HPP
#define MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_TYPE_HPP

#include "binding_info.hpp"

#include <mlpack/bindings/cli/get_printable_type.hpp>
#include <mlpack/bindings/python/get_printable_type.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Print the type of a parameter into the output string.  The type printed
 * depends on the current setting of BindingInfo::Language().
 */
template<typename T>
void GetPrintableType(const util::ParamData& data,
                      const void* /* input */,
                      void* output)
{
  if (BindingInfo::Language() == "cli")
  {
    *((std::string*) output) =
        cli::GetPrintableType<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "python")
  {
    *((std::string*) output) =
        python::GetPrintableType<typename std::remove_pointer<T>::type>(data);
  }
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
