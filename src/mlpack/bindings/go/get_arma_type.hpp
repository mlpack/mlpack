/**
 * @file get_arma_type.hpp
 * @author Yasmine Dumouchle
 *
 * Given a matrix type, return the letter we should append to get the right
 * ArmaToGonum or GonumToArma method call.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_GET_ARMA_TYPE_HPP
#define MLPACK_BINDINGS_GO_GET_ARMA_TYPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace go {

template<typename T>
inline std::string GetArmaType()
{
  return "";
}

template<>
inline std::string GetArmaType<arma::Mat<size_t>>()
{
  return "u";
}

template<>
inline std::string GetArmaType<arma::Col<size_t>>()
{
  return "u";
}

template<>
inline std::string GetArmaType<arma::Row<size_t>>()
{
  return "u";
}

// double = d.
template<>
inline std::string GetArmaType<arma::mat>()
{
  return "";
}

template<>
inline std::string GetArmaType<arma::vec>()
{
  return "";
}

template<>
inline std::string GetArmaType<arma::rowvec>()
{
  return "";
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
