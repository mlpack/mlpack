/**
 * @file print_docs.hpp
 * @author Ryan Curtin
 *
 * Functions to generate Markdown for documentation of bindings.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_DOCS_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_DOCS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Given the current settings of CLI, print the header (which will be the
 * navigation tab) for the binding types that are registered for these options.
 *
 * Output is printed to stdout.
 */
void PrintHeaders(const std::string& bindingName,
                  const std::vector<std::string>& languages);

/**
 * Given the current settings of CLI, print Markdown documentation for the
 * binding types that are registered for these options.
 *
 * Output is printed to stdout.
 *
 * @param bindingName The binding name (${BINDING} from CMake).
 * @param languages The set of languages to print documentation for.
 */
void PrintDocs(const std::string& bindingName,
               const std::vector<std::string>& languages);

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
