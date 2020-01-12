/**
 * @file get_printable_type.hpp
 * @author Ryan Curtin
 *
 * Get the printable type of a parameter.  This type is not the C++ type but
 * instead the Julia type that a user would use.
 */
#ifndef MLPACK_BINDINGS_JULIA_GET_PRINTABLE_TYPE_HPP
#define MLPACK_BINDINGS_JULIA_GET_PRINTABLE_TYPE_HPP

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0);

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0);

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string GetPrintableType(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Print the command-line type of an option into a string.
 */
template<typename T>
void GetPrintableType(const util::ParamData& data,
                       const void* /* input */,
                       void* output)
{
  *((std::string*) output) =
      GetPrintableType<typename std::remove_pointer<T>::type>(data);
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#include "get_printable_type_impl.hpp"

#endif
