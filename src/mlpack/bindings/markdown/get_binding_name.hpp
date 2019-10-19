/**
 * @file get_binding_name.cpp
 * @author Ryan Curtin
 *
 * Given the name of a binding as it appears in CMake, return the corresponding
 * name of the binding that is generated for a given language.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_GET_BINDING_NAME_HPP
#define MLPACK_BINDINGS_MARKDOWN_GET_BINDING_NAME_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Given a language name and a binding name, return the name of that binding for
 * that language.  Note that if a new language is added to the mlpack bindings,
 * this method will need to be updated so that documentation can be successfully
 * generated for that language.
 */
std::string GetBindingName(const std::string& language,
                           const std::string& name);

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
