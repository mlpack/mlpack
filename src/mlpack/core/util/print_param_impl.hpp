/**
 * @file print_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of parameter printing functions.
 */
#ifndef MLPACK_CORE_UTIL_PRINT_PARAM_IMPL_HPP
#define MLPACK_CORE_UTIL_PRINT_PARAM_IMPL_HPP

#include "print_param.hpp"

namespace mlpack {
namespace util {

//! Print an option.
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<IsStdVector<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  Log::Info << boost::any_cast<T>(data.value);
}

//! Print a vector option.
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::enable_if<IsStdVector<T>>::type* /* junk */)
{
  const T& t = boost::any_cast<T>(data.value);

  for (size_t i = 0; i < t.size(); ++i)
    Log::Info << t[i] << " ";
}

//! Print a matrix option (this just prints the filename).
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  Log::Info << boost::any_cast<std::string>(data.value);
}

//! Print a model option (this just prints the filename).
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  Log::Info << boost::any_cast<std::string>(data.value);
}

//! Print a mapped matrix option (this just prints the filename).
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  Log::Info << boost::any_cast<std::string>(data.value);
}

} // namespace util
} // namespace mlpack

#endif
