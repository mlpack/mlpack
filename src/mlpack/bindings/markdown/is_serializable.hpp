/**
 * @file is_serializable.hpp
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
 * Return false, because the type is not serializable.
 */
template<typename T>
bool IsSerializable(
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
{
  return false;
}

/**
 * Return false, because even though the type is serializable, it is an
 * Armadillo type not an mlpack model.
 */
template<typename T>
bool IsSerializable(
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  return false;
}

/**
 * Return true, because the type is serializable.
 */
template<typename T>
bool IsSerializable(
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0)
{
  return true;
}

/**
 * Return whether or not the type is serializable.
 */
template<typename T>
void IsSerializable(const util::ParamData& /* data */,
                    const void* /* input */,
                    void* output)
{
  *((bool*) output) = IsSerializable<typename std::remove_pointer<T>::type>();
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
