/**
 * @file bindings/julia/print_model_type_import.hpp
 * @author Ryan Curtin
 *
 * If the type is serializable, we need to define a special utility function to
 * set a IO parameter of that type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_MODEL_TYPE_IMPORT_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_MODEL_TYPE_IMPORT_HPP

#include <mlpack/bindings/util/strip_type.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * If the type is not serializable, print nothing.
 */
template<typename T>
void PrintModelTypeImport(
    util::ParamData& /* d */,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0)
{
  // Do nothing.
}

/**
 * Matrices are serializable but here we also print nothing.
 */
template<typename T>
void PrintModelTypeImport(
    util::ParamData& /* d */,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  // Do nothing.
}

/**
 * For non-matrix serializable types we need to print something.
 */
template<typename T>
void PrintModelTypeImport(
    util::ParamData& d,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  // We need to print, e.g.,
  // import ..<type>

  // First, print the struct definition.
  std::cout << "import .." << util::StripType(d.cppType) << std::endl;
}

/**
 * If the type is serializable, print the definition of a special utility
 * function to set a IO parameter of that type to stdout.
 */
template<typename T>
void PrintModelTypeImport(util::ParamData& d,
                          const void* /* input */,
                          void* /* output */)
{
  PrintModelTypeImport<std::remove_pointer_t<T>>(d);
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
