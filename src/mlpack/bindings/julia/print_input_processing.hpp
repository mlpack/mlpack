/**
 * @file print_input_processing.hpp
 * @author Ryan Curtin
 *
 * Print Julia code to handle input arguments.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_INPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_INPUT_PROCESSING_HPP

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the input processing (basically calling CLI::GetParam<>()) for a
 * non-serializable type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const std::string& functionName,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Print the input processing for an Armadillo type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const std::string& functionName,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Print the input processing for a serializable type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const std::string& functionName,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Print the input processing (basically calling CLI::GetParam<>()) for a
 * matrix with DatasetInfo type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const std::string& functionName,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Print the input processing (basically calling CLI::GetParam<>()) for a type.
 */
template<typename T>
void PrintInputProcessing(const util::ParamData& d,
                          const void* input,
                          void* /* output */)
{
  // Call out to the right overload.
  PrintInputProcessing<typename std::remove_pointer<T>::type>(d,
      *((std::string*) input));
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#include "print_input_processing_impl.hpp"

#endif
