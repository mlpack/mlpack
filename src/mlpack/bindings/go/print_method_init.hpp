/**
 * @file print_method_init.hpp
 * @author Yasmine Dumouchel
 *
 * Print method initiation function for a Go Binding
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_METHOD_INIT_HPP
#define MLPACK_BINDINGS_GO_PRINT_METHOD_INIT_HPP

#include <mlpack/prereqs.hpp>
#include "get_go_type.hpp"
#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print parameter with it default value for a standard option type.
 */
template<typename T>
void PrintMethodInit(
    const util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "nil";
  if (std::is_same<T, bool>::value)
    def = "false";

    std::string name = d.name;
    std::string goParamName = name;
    if (!name.empty())
    {
      goParamName[0] = std::toupper(goParamName[0]);
    }

  /**
   * This gives us code like:
   *
   */
  if (!d.required)
  {

    if (d.cppType == "std::string")
    {
      std::string value = boost::any_cast<std::string>(d.value);
      std::cout << prefix << goParamName << ": \"" << value << "\"," << std::endl;
    }
    else if (d.cppType == "double")
    {
      double value = boost::any_cast<double>(d.value);
      std::cout << prefix << goParamName << ": " << value << "," << std::endl;
    }
    else if (d.cppType == "int")
    {
      int value = boost::any_cast<int>(d.value);
      std::cout << prefix << goParamName << ": " << value << "," << std::endl;
    }
    else if (d.cppType == "bool")
    {
      bool value = boost::any_cast<bool>(d.value);
      if (value == 0)
        std::cout << prefix << goParamName << ": false," << std::endl;
      else
        std::cout << prefix << goParamName << ": true," << std::endl;
    }
  }
}

/**
 * Print parameter with it default value for a matrix type.
 */
template<typename T>
void PrintMethodInit(
    const util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "nil";
  if (std::is_same<T, bool>::value)
    def = "false";

    // Make sure that we don't use names that are Python keywords.
    std::string name = d.name;
    std::string goParamName = name;
    if (!name.empty())
    {
      goParamName[0] = std::toupper(goParamName[0]);
    }

  /**
   * This gives us code like:
   *
   */
  if (!d.required)
  {
    std::cout << prefix << goParamName << ": " << def << ","
        << std::endl;
  }
}

/**
 * Print parameter with it default valuefor a serializable type.
 */
template<typename T>
void PrintMethodInit(
    const util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "nil";
  if (std::is_same<T, bool>::value)
    def = "false";

    // Make sure that we don't use names that are Python keywords.
    std::string name = d.name;
    std::string goParamName = name;
    if (!name.empty())
    {
      goParamName[0] = std::toupper(goParamName[0]);
    }

  /**
   * This gives us code like:
   *
   */
  std::cout << prefix << goParamName << ": " << def << ","
        << std::endl;

}

// /**
//  * Print input processing for a matrix/DatasetInfo type.
//  */
// template<typename T>
// void PrintMethodInit(
//     const util::ParamData& d,
//     const size_t indent,
//     const typename boost::enable_if<std::is_same<T,
//         std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
// {
// }

/**
 * Given parameter information and the current number of spaces for indentation,
 * print the code to process the input to cout.  This code assumes that
 * data.input is true, and should not be called when data.input is false.
 *
 * The number of spaces to indent should be passed through the input pointer.
 *
 * @param d Parameter data struct.
 * @param input Pointer to size_t holding the indentation.
 * @param output Unused parameter.
 */
template<typename T>
void PrintMethodInit(const util::ParamData& d,
                          const void* input,
                          void* /* output */)
{
  PrintMethodInit<typename std::remove_pointer<T>::type>(d,
      *((size_t*) input));
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
