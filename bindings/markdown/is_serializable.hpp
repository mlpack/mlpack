/**
 * @file bindings/markdown/is_serializable.hpp
 * @author Ryan Curtin
 *
 * Return a bool noting whether or not a parameter is serializable.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_IS_SERIALIZABLE_HPP
#define MLPACK_BINDINGS_MARKDOWN_IS_SERIALIZABLE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Return false, because the type is not serializable.  This includes Armadillo
 * types, which we say aren't serializable (in this context) because they aren't
 * mlpack models.
 */
template<typename T>
bool IsSerializable(
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
{
  return false;
}

/**
 * Return true, because the type is serializable.
 */
template<typename T>
bool IsSerializable(
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0)
{
  return true;
}

/**
 * Return whether or not the type is serializable.
 */
template<typename T>
void IsSerializable(util::ParamData& /* data */,
                    const void* /* input */,
                    void* output)
{
  *((bool*) output) = IsSerializable<typename std::remove_pointer<T>::type>();
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
