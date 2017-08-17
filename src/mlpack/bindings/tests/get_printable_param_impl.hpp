/**
 * @file get_printable_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of parameter printing functions.
 */
#ifndef MLPACK_BINDINGS_TESTS_GET_PRINTABLE_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_TESTS_GET_PRINTABLE_PARAM_IMPL_HPP

#include "get_printable_param.hpp"

namespace mlpack {
namespace bindings {
namespace tests {

//! Print an option.
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<util::IsStdVector<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  std::ostringstream oss;
  oss << boost::any_cast<T>(data.value);
  return oss.str();
}

//! Print a vector option.
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& data,
    const typename boost::enable_if<util::IsStdVector<T>>::type* /* junk */)
{
  const T& t = boost::any_cast<T>(data.value);

  std::ostringstream oss;
  for (size_t i = 0; i < t.size(); ++i)
    oss << t[i] << " ";
  return oss.str();
}

//! Print a matrix option (this just prints 'matrix type').
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& /* data */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  return "matrix type";
}

//! Print a model option (this just prints the filename).
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& data,
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  // Extract the string from the tuple that's being held.
  std::ostringstream oss;
  oss << data.cppType << " model";
  return oss.str();
}

//! Print a mapped matrix option (this just prints 'matrix/DatasetInfo tuple').
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& /* data */,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  return "matrix/DatatsetInfo tuple";
}

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
