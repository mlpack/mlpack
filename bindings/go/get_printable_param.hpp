/**
 * @file bindings/go/get_printable_param.hpp
 * @author Yashwant Singh
 *
 * Get a printable version of parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_GET_PRINTABLE_PARAM_HPP
#define MLPACK_BINDINGS_GO_GET_PRINTABLE_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print an option of a simple type.
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  std::ostringstream oss;
  oss << std::any_cast<T>(data.value);
  return oss.str();
}

/**
 * Print a vector option, with spaces between it.
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0)
{
  const T& t = std::any_cast<T>(data.value);

  std::ostringstream oss;
  for (size_t i = 0; i < t.size(); ++i)
    oss << t[i] << " ";
  return oss.str();
}

/**
 * Print a matrix option (this prints its size).
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // Get the matrix.
  const T& matrix = std::any_cast<T>(data.value);

  std::ostringstream oss;
  oss << matrix.n_rows << "x" << matrix.n_cols << " matrix";
  return oss.str();
}

/**
 * Print a serializable class option (this prints the class name).
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  std::ostringstream oss;
  oss << data.cppType << " model at " << std::any_cast<T*>(data.value);
  return oss.str();
}

/**
 * Print a combination DatasetInfo/matrix parameter.
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  // Get the matrix.
  const T& tuple = std::any_cast<T>(data.value);
  const arma::mat& matrix = std::get<1>(tuple);

  std::ostringstream oss;
  oss << matrix.n_rows << "x" << matrix.n_cols << " matrix with dimension type "
      << "information";
  return oss.str();
}

/**
 * Print an option into a std::string.  This should print a short, one-line
 * representation of the object.  The string will be stored in the output
 * pointer.
 *
 * @param data Parameter data struct.
 * @param * (input) Unused parameter.
 * @param output Output storage for the string.
 */
template<typename T>
void GetPrintableParam(util::ParamData& data,
                       const void* /* input */,
                       void* output)
{
  *((std::string*) output) =
      GetPrintableParam<typename std::remove_pointer<T>::type>(data);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
