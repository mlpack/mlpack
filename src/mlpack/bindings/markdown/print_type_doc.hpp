/**
 * @file print_type_doc.hpp
 * @author Ryan Curtin
 *
 * Print documentation for a given type, depending on the current language (as
 * set in BindingInfo).
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_TYPE_DOC_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_TYPE_DOC_HPP

#include "binding_info.hpp"

#include <mlpack/bindings/cli/print_type_doc.hpp>
#include <mlpack/bindings/python/print_type_doc.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Print the type of a parameter into the output string.  The type printed
 * depends on the current setting of BindingInfo::Language().
 */
template<typename T>
std::string PrintTypeDoc(const util::ParamData& data)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
  }
  else
  {
    throw std::invalid_argument("PrintTypeDoc(): unknown "
        "BindingInfo::Language()" + BindingInfo::Language() + "!");
  }
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
