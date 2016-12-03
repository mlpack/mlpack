/**
 * @file output_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementations of functions to output parameters of different types.
 */
#ifndef MLPACK_CORE_UTIL_OUTPUT_PARAM_IMPL_HPP
#define MLPACK_CORE_UTIL_OUTPUT_PARAM_IMPL_HPP

#include "output_param.hpp"
#include <mlpack/core/data/save.hpp>

namespace mlpack {
namespace util {

//! Output an option.
template<typename T>
void OutputParamImpl(
    const ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<IsStdVector<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  std::cout << data.name << ": " << *boost::any_cast<T>(&data.value)
      << std::endl;
}

//! Output a vector option.
template<typename T>
void OutputParamImpl(
    const ParamData& data,
    const typename boost::enable_if<IsStdVector<T>>::type* /* junk */)
{
  std::cout << data.name << ": ";
  const T& t = *boost::any_cast<T>(&data.value);
  for (size_t i = 0; i < t.size(); ++i)
    std::cout << t[i] << " ";
  std::cout << std::endl;
}

//! Output a matrix option (this saves it to file).
template<typename T>
void OutputParamImpl(
    const ParamData& data,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  const T& output = *boost::any_cast<T>(&data.mappedValue);
  const std::string& filename = *boost::any_cast<std::string>(&data.value);

  if (output.n_elem > 0 && filename != "")
    data::Save(filename, output, false, !data.noTranspose);
}

//! Output a model option (this saves it to file).
template<typename T>
void OutputParamImpl(
    const ParamData& data,
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  // The const cast is necessary here because Serialize() can't ever be marked
  // const.  In this case we can assume it though, since we will be saving and
  // not loading.
  T& output = const_cast<T&>(*boost::any_cast<T>(&data.mappedValue));
  const std::string& filename = *boost::any_cast<std::string>(&data.value);

  if (filename != "")
    data::Save(filename, "model", output);
}

//! Output a mapped dataset.
template<typename T>
void OutputParamImpl(
    const ParamData& data,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  // Output the matrix with the mappings.
//  const data::DatasetInfo& info =
//      std::get<0>(*boost::any_cast<T>(&data.mappedValue));
  const arma::mat& matrix = std::get<1>(*boost::any_cast<T>(&data.mappedValue));
  const std::string& filename = *boost::any_cast<std::string>(&data.value);

  // The mapping isn't taken into account.  We should write a data::Save()
  // overload for this.
  if (filename != "")
    data::Save(filename, matrix, false, !data.noTranspose);
}

} // namespace util
} // namespace mlpack

#endif
