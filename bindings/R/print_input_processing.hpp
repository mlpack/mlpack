/**
 * @file bindings/R/print_input_processing.hpp
 * @author Yashwant Singh Parihar
 *
 * Print R code to handle input arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_INPUT_PROCESSING_IMPL_HPP
#define MLPACK_BINDINGS_R_PRINT_INPUT_PROCESSING_IMPL_HPP

#include <mlpack/bindings/util/strip_type.hpp>
#include "get_type.hpp"
#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Print input processing for a standard option type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  if (!d.required)
  {
    /**
     * This gives us code like:
     *
     *     if (!identical(<param_name>, NA)) {
     *        SetParam<type>(p, "<param_name>", <param_name>)
     *     }
     */
    MLPACK_COUT_STREAM << "  if (!identical(" << d.name;
    if (d.cppType == "bool")
    {
      MLPACK_COUT_STREAM << ", FALSE)) {" << std::endl;
    }
    else
    {
      MLPACK_COUT_STREAM << ", NA)) {" << std::endl;
    }
    MLPACK_COUT_STREAM << "    SetParam" << GetType<T>(d) << "(p, \""
        << d.name << "\", " << d.name << ")" << std::endl;
    MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
  }
  else
  {
    /**
     * This gives us code like:
     *
     *     SetParam<type>(p, "<param_name>", <param_name>)
     */
    MLPACK_COUT_STREAM << "  SetParam" << GetType<T>(d) << "(p, \""
              << d.name << "\", " << d.name << ")" << std::endl;
  }
  MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a matrix type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  std::string extraTransStr = "";
  if (d.cppType == "arma::mat")
  {
    if (d.noTranspose)
      extraTransStr = ", FALSE";
    else
      extraTransStr = ", TRUE";
  }

  if (!d.required)
  {
    /**
     * This gives us code like:
     *
     *     if (!identical(<param_name>, NA)) {
     *        SetParam<type>(p, "<param_name>", to_matrix(<param_name>))
     *     }
     *
     * and if the parameter is an arma::mat, we will get code like
     *
     *     if (!identical(<param_name>, NA)) {
     *        SetParam<type>(p, "<param_name>", to_matrix(<param_name>), TRUE)
     *     }
     *
     * where the final boolean specifies whether the matrix should be
     * transposed.
     */
    MLPACK_COUT_STREAM << "  if (!identical(" << d.name << ", NA)) {"
        << std::endl;
    MLPACK_COUT_STREAM << "    SetParam" << GetType<T>(d) << "(p, \""
        << d.name << "\", to_matrix(" << d.name << ")" << extraTransStr << ")"
        << std::endl;
    MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
  }
  else
  {
    /**
     * This gives us code like:
     *
     *     SetParam<type>(p, "<param_name>", to_matrix(<param_name>))
     *
     * and if the parameter is an arma::mat, we will get code like
     *
     *     SetParam<type>(p, "<param_name>", to_matrix(<param_name>), TRUE)
     *
     * where the final boolean specifies whether the matrix should be
     * transposed.
     */
    MLPACK_COUT_STREAM << "  SetParam" << GetType<T>(d) << "(p, \""
        << d.name << "\", to_matrix(" << d.name << ")" << extraTransStr << ")"
        << std::endl;
  }
  MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a matrix with info type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  if (!d.required)
  {
    /**
     * This gives us code like:
     *
     *     if (!identical(<param_name>, NA)) {
     *        <param_name> = to_matrix_with_info(<param_name>)
     *        SetParam<type>(p, "<param_name>", <param_name>$info,
     *                       <param_name>$data)
     *     }
     */
    MLPACK_COUT_STREAM << "  if (!identical(" << d.name << ", NA)) {"
        << std::endl;
    MLPACK_COUT_STREAM << "    " << d.name << " <- to_matrix_with_info("
        << d.name << ")" << std::endl;
    MLPACK_COUT_STREAM << "    SetParam" << GetType<T>(d) << "(p, \""
        << d.name << "\", " << d.name << "$info, " << d.name
        << "$data)" << std::endl;
    MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
  }
  else
  {
    /**
     * This gives us code like:
     *
     *     <param_name> = to_matrix_with_info(<param_name>)
     *     SetParam<type>(p, "<param_name>", <param_name>$info,
     *                    <param_name>$data)
     */
    MLPACK_COUT_STREAM << "  " << d.name << " <- to_matrix_with_info("
        << d.name << ")" << std::endl;
    MLPACK_COUT_STREAM << "  SetParam" << GetType<T>(d) << "(p, \""
        << d.name << "\", " << d.name << "$info, " << d.name
        << "$data)" << std::endl;
  }
  MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a serializable type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  if (!d.required)
  {
    /**
     * This gives us code like:
     *
     *     if (!identical(<param_name>, NA)) {
     *        SetParam<ModelType>Ptr(p, "<param_name>", <param_name>)
     *        # Add to the list of input models we received.
     *        inputModels <- append(inputModels, <param_name>)
     *     }
     */
    MLPACK_COUT_STREAM << "  if (!identical(" << d.name << ", NA)) {"
        << std::endl;
    MLPACK_COUT_STREAM << "    SetParam" << util::StripType(d.cppType)
        << "Ptr(p, \"" << d.name << "\", " << d.name << ")" << std::endl;
    MLPACK_COUT_STREAM << "    # Add to the list of input models we received."
        << std::endl;
    MLPACK_COUT_STREAM << "    inputModels <- append(inputModels, " << d.name
        << ")" << std::endl;
    MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
  }
  else
  {
    /**
     * This gives us code like:
     *
     *     SetParam<ModelType>Ptr(p, "<param_name>", <param_name>)
     */
    MLPACK_COUT_STREAM << "  SetParam" << util::StripType(d.cppType)
        << "Ptr(p, \"" << d.name << "\", " << d.name << ")" << std::endl;
  }
  MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * @param d Parameter data struct.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintInputProcessing(util::ParamData& d,
                          const void* /* input */,
                          void* /* output */)
{
  PrintInputProcessing<typename std::remove_pointer<T>::type>(d);
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
