/**
 * @file bindings/cli/in_place_copy.hpp
 * @author Ryan Curtin
 *
 * Use template metaprogramming to set filenames correctly for in-place copy
 * arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_IN_PLACE_COPY_HPP
#define MLPACK_BINDINGS_CLI_IN_PLACE_COPY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * This overload is called when nothing special needs to happen to make
 * something an in-place copy.
 *
 * @param * (d) ParamData object to get parameter value from. (Unused.)
 * @param * (input) ParamData object that represents the option we are making a copy
 *     of. (Unused.)
 */
template<typename T>
void InPlaceCopyInternal(
    util::ParamData& /* d */,
    util::ParamData& /* input */,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>* = 0)
{
  // Nothing to do.
}

/**
 * Modify the filename for any type that needs to be loaded from disk to match
 * the filename of the input parameter, for a matrix/DatasetInfo parameter.
 *
 * @param d ParamData object we want to make into an in-place copy.
 * @param input ParamData object whose filename we should copy.
 */
template<typename T>
void InPlaceCopyInternal(
    util::ParamData& d,
    util::ParamData& input,
    const std::enable_if_t<
        arma::is_arma_type<T>::value ||
        std::is_same_v<T, std::tuple<mlpack::data::DatasetInfo, arma::mat>>>*
            = 0)
{
  // Make the output filename the same as the input filename.
  using TupleType = std::tuple<T, typename ParameterType<T>::type>;
  TupleType& tuple = *std::any_cast<TupleType>(&d.value);
  std::string& value = std::get<0>(std::get<1>(tuple));

  const TupleType& inputTuple = *std::any_cast<TupleType>(&input.value);
  value = std::get<0>(std::get<1>(inputTuple));
}

/**
 * Modify the filename for any type that needs to be loaded from disk to match
 * the filename of the input parameter. For serializable objects.
 *
 * @param d ParamData object we want to make into an in-place copy.
 * @param input ParamData object whose filename we should copy.
 */
template<typename T>
void InPlaceCopyInternal(
    util::ParamData& d,
    util::ParamData& input,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  // Make the output filename the same as the input filename.
  using TupleType = std::tuple<T*, typename ParameterType<T>::type>;
  TupleType& tuple = *std::any_cast<TupleType>(&d.value);
  std::string& value = std::get<1>(tuple);

  const TupleType& inputTuple = *std::any_cast<TupleType>(&input.value);
  value = std::get<1>(inputTuple);
}

/**
 * Make the given ParamData be an in-place copy of the input.
 *
 * @param d Parameter information.
 * @param input Input ParamData we would like be the source of the in-place
 *      copy.
 * @param * (output) Unused parameter.
 */
template<typename T>
void InPlaceCopy(util::ParamData& d,
                 const void* input,
                 void* /* output */)
{
  // Cast to the correct type.
  InPlaceCopyInternal<std::remove_pointer_t<T>>(
      const_cast<util::ParamData&>(d), *((util::ParamData*) input));
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
