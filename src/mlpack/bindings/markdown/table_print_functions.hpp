/**
 * @file print_table_functions.hpp
 * @author Ryan Curtin
 *
 * This should be renamed.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_TABLE_FUNCTIONS_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_TABLE_FUNCTIONS_HPP

namespace mlpack {
namespace bindings {
namespace markdown {

std::string ParamString(const std::string& name);

std::string ParamType(const std::string& type);

std::string ParamDescription(const std::string& desc);

std::string ParamDefault(const std::string& def);

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
