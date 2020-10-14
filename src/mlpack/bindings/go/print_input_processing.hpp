/**
 * @file bindings/go/print_input_processing.hpp
 * @author Yasmine Dumouchel
 *
 * Print input processing for a Go binding option.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_INPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_GO_PRINT_INPUT_PROCESSING_HPP

#include <mlpack/prereqs.hpp>
#include "get_type.hpp"
#include "get_go_type.hpp"
#include "strip_type.hpp"
#include <mlpack/bindings/util/camel_case.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print input processing for a standard option type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
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

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string paramName = d.name;
  std::string goParamName = paramName;
  if (!paramName.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  /**
   * This gives us code like:
   *
   *  // Detect if the parameter was passed; set if so.
   *  if param.Name != nil {
   *     setParam<d.cppType>("paramName", param.Name)
   *     setPassed("paramName")
   *  }
   */
  std::cout << prefix << "// Detect if the parameter was passed; set if so."
            << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if param." << goParamName << " != ";

    // Print out default value.
    if (d.cppType == "std::string")
    {
      std::string value = boost::any_cast<std::string>(d.value);
      std::cout << "\"" << value << "\"";
    }
    else if (d.cppType == "double")
    {
      double value = boost::any_cast<double>(d.value);
      std::cout << value;
    }
    else if (d.cppType == "int")
    {
      int value = boost::any_cast<int>(d.value);
      std::cout << value;
    }
    else if (d.cppType == "bool")
    {
      bool value = boost::any_cast<bool>(d.value);
      if (value == 0)
      std::cout << "false";
      else
      std::cout << "true";
    }
    else if (GetType<T>(d) == "VecString" || GetType<T>(d) == "VecInt")
    {
      std::cout << "nil";
    }

    // Print function call to set the given parameter into the io.
    std::cout << " {" << std::endl;
    std::cout << prefix << prefix << "setParam" << GetType<T>(d) << "(\""
              << d.name << "\", param." << goParamName << ")" << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << prefix << "setPassed(\""
              << d.name << "\")" << std::endl;

    // If this parameter is "verbose", then enable verbose output.
    if (d.name == "verbose")
      std::cout << prefix << prefix << "enableVerbose()" << std::endl;

    std::cout << prefix << "}" << std::endl; // Closing brace.
  }
  else
  {
    goParamName = util::CamelCase(goParamName, true);
    // Print function call to set the given parameter into the io.
    std::cout << prefix << "setParam" << GetType<T>(d) << "(\""
              << d.name << "\", " << goParamName << ")"
              << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << "setPassed(\"" << d.name << "\")" << std::endl;
  }
  std::cout << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a matrix type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string paramName = d.name;
  std::string goParamName = paramName;
  if (!paramName.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  /**
   * This gives us code like:
   *
   *  // Detect if the parameter was passed; set if so.
   *  if param.Name != nil {
   *     gonumToArma<type>("paramName", param.Name)
   *     setPassed("paramName")
   *  }
   */
  std::cout << prefix << "// Detect if the parameter was passed; set if so."
            << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if param." << goParamName
              << " != nil {" << std::endl;

    // Print function call to set the given parameter into the io.
    std::cout << prefix << prefix << "gonumToArma" << GetType<T>(d)
              << "(\"" << d.name << "\", param." << goParamName
              << ")" << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << prefix << "setPassed(\"" << d.name << "\")"
              << std::endl;
    std::cout << prefix << "}" << std::endl; // Closing brace.
  }
  else
  {
    goParamName = util::CamelCase(goParamName, true);
    // Print function call to set the given parameter into the io.
    std::cout << prefix << "gonumToArma" << GetType<T>(d)
              << "(\"" << d.name << "\", " << goParamName
              << ")" << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << "setPassed(\"" << d.name << "\")" << std::endl;
  }
  std::cout << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a matrix with info type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string paramName = d.name;
  std::string goParamName = paramName;
  if (!paramName.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  /**
   * This gives us code like:
   *
   *  // Detect if the parameter was passed; set if so.
   *  if param.Name != nil {
   *     gonumToArmaMatWithInfo<type>("paramName", param.Name)
   *     setPassed("paramName")
   *  }
   */
  std::cout << prefix << "// Detect if the parameter was passed; set if so."
            << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if param." << goParamName
              << " != nil {" << std::endl;

    // Print function call to set the given parameter into the io.
    std::cout << prefix << prefix << "gonumToArmaMatWithInfo"
              << "(\"" << d.name << "\", param." << goParamName
              << ")" << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << prefix << "setPassed(\"" << d.name << "\")"
              << std::endl;
    std::cout << prefix << "}" << std::endl; // Closing brace.
  }
  else
  {
    goParamName = util::CamelCase(goParamName, true);
    // Print function call to set the given parameter into the io.
    std::cout << prefix << "gonumToArmaMatWithInfo"
              << "(\"" << d.name << "\", " << goParamName
              << ")" << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << "setPassed(\"" << d.name << "\")" << std::endl;
  }
  std::cout << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a serializable type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // First, get the correct classparamName if needed.
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);

  const std::string prefix(indent, ' ');

  // Capitalize the first letter of parameter name so it is
  // of exported type in Go.
  std::string paramName = d.name;
  std::string goParamName = paramName;
  if (!paramName.empty())
  {
    goParamName = util::CamelCase(goParamName, false);
  }

  /**
   * This gives us code like:
   *
   *  // Detect if the parameter was passed; set if so.
   *  if param.Name != nil {
   *     set<ModelType>("paramName", param.Name)
   *     setPassed("paramName")
   *  }
   */
  std::cout << prefix << "// Detect if the parameter was passed; set if so."
      << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if param." << goParamName << " != nil {"
              << std::endl;
    // Print function call to set the given parameter into the io.
    std::cout << prefix << prefix << "set" << strippedType << "(\""
              << d.name << "\", param." << goParamName << ")" << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << prefix << "setPassed(\"" << d.name << "\")"
              << std::endl;
    std::cout << prefix << "}" << std::endl; // Closing brace.
  }
  else
  {
    goParamName = util::CamelCase(goParamName, true);
    // Print function call to set the given parameter into the io.
    std::cout << prefix << "set" << strippedType << "(\"" << d.name
              << "\", " << goParamName << ")" << std::endl;

    // Print function call to set the given parameter as passed.
    std::cout << prefix << "setPassed(\"" << d.name << "\")" << std::endl;
  }
  std::cout << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Given parameter information and the current number of spaces for indentation,
 * print the code to process the input to cout.  This code assumes that
 * data.input is true, and should not be called when data.input is false.
 *
 * The number of spaces to indent should be passed through the input pointer.
 *
 * @param d Parameter data struct.
 * @param input Pointer to size_t holding the indentation.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintInputProcessing(util::ParamData& d,
                          const void* input,
                          void* /* output */)
{
  PrintInputProcessing<typename std::remove_pointer<T>::type>(d,
      *((size_t*) input));
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
