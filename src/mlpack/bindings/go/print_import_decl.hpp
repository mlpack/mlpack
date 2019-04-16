/**
 * @file import_decl.hpp
 * @author Yasmine Dumouchel
 *
 * Print the necessary imports for go bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_IMPORT_DECL_HPP
#define MLPACK_BINDINGS_GO_IMPORT_DECL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * For a serializable type, print a cppclass definition.
 */
template<typename T>
void ImportDecl(
    const util::ParamData& /* d */,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  /**
   * This will give output of the form:
   *
   */
  const std::string prefix = std::string(indent, ' ');
  // Now import all the necessary packages.
  std::cout << prefix << "\"runtime\" " << std::endl;
  std::cout << prefix << "\"unsafe\" " << std::endl;
}

/**
 * For a non-serializable type, print nothing.
 */
template<typename T>
void ImportDecl(
    const util::ParamData& /* d */,
    const size_t /* indent */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
{
  // Print nothing.
}

/**
 * For a matrix type, print nothing.
 */
template<typename T>
void ImportDecl(
    const util::ParamData& /* d */,
    const size_t /* indent */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // Print nothing.
}

/**
 * Print the cppclass definition for a serializable model; print nothing for a
 * non-serializable type.
 *
 * @param d Parameter info struct.
 * @param input Pointer to size_t indicating indent.
 * @param output Unused parameter.
 */
template<typename T>
void ImportDecl(const util::ParamData& d,
                const void* indent,
                void* /* output */)
{
  ImportDecl<typename std::remove_pointer<T>::type>(d, *((size_t*) indent));
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
