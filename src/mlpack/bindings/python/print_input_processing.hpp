/**
 * @file print_input_processing.hpp
 * @author Ryan Curtin
 *
 * Print input processing for a Python binding option.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_INPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_INPUT_PROCESSING_HPP

#include <mlpack/prereqs.hpp>
#include "get_numpy_type_char.hpp"
#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Print input processing for a standard option type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  std::string def = "None";
  if (std::is_same<T, bool>::value)
    def = "False";

  // Make sure that we don't use names that are Python keywords.
  std::string name = (d.name == "lambda") ? "lambda_" : d.name;

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   SetParam[int](<const string> 'param_name', param_name)
   *   CLI.SetPassed(<const string> 'param_name')
   */
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if " << name << " is not " << def << ":"
        << std::endl;

    std::cout << prefix << "  SetParam[" << GetPythonType<T>(d) << "](<const "
        << "string> '" << d.name << "', " << name << ")" << std::endl;
    std::cout << prefix << "  CLI.SetPassed(<const string> '" << d.name
        << "')" << std::endl;

    // If this parameter is "verbose", then enable verbose output.
    if (d.name == "verbose")
      std::cout << prefix << "  EnableVerbose()" << std::endl;
  }
  else
  {
    std::cout << prefix << "SetParam[" << GetPythonType<T>(d) << "](<const "
        << "string> '" << d.name << "', " << name << ")" << std::endl;
    std::cout << prefix << "CLI.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
  }
  std::cout << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a matrix type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   param_name_mat = arma_numpy.numpy_to_mat_d(param_name)
   *   SetParam[mat](<const string> 'param_name', dereference(param_name_mat))
   *   CLI.SetPassed(<const string> 'param_name')
   */
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if " << d.name << " is not None:" << std::endl;

    std::cout << prefix << "  " << d.name << "_mat = arma_numpy.numpy_to_mat_"
        << GetNumpyTypeChar<T>() << "(to_matrix(" << d.name << "))"
        << std::endl;
    std::cout << prefix << "  SetParam[" << GetPythonType<T>(d) << "](<const "
        << "string> '" << d.name << "', dereference(" << d.name << "_mat))"
        << std::endl;
    std::cout << prefix << "  CLI.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
  }
  else
  {
    std::cout << prefix << d.name << "_mat = arma_numpy.numpy_to_mat_"
        << GetNumpyTypeChar<T>() << "(to_matrix(" << d.name << "))"
        << std::endl;
    std::cout << prefix << "SetParam[" << GetPythonType<T>(d) << "](<const "
        << "string> '" << d.name << "', dereference(" << d.name << "_mat))"
        << std::endl;
    std::cout << prefix << "CLI.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
  }
  std::cout << std::endl;
}

/**
 * Print input processing for a serializable type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // First, get the correct class name if needed.
  std::string strippedType, printedType, defaultsType;
  StripType(d.cppType, strippedType, printedType, defaultsType);

  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   MoveFromPtr[Model](CLI.GetParam[Model]('param_name'),
   *       (<ModelType?> param_name).modelptr)
   *   CLI.SetPassed(<const string> 'param_name')
   */
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if " << d.name << " is not None:" << std::endl;
    std::cout << prefix << "  MoveFromPtr[" << strippedType << "](CLI.GetParam["
        << strippedType << "]('" << d.name << "'), (<" << strippedType
        << "Type?> " << d.name << ").modelptr)" << std::endl;
    std::cout << prefix << "  CLI.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
  }
  else
  {
    std::cout << prefix << "MoveFromPtr[" << strippedType << "](CLI.GetParam["
        << strippedType << "]('" << d.name << "'), (<" << strippedType
        << "Type?> " << d.name << ").modelptr)" << std::endl;
    std::cout << prefix << "CLI.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
  }
  std::cout << std::endl;
}

/**
 * Print input processing for a matrix/DatasetInfo type.
 */
template<typename T>
void PrintInputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // TODO: figure this out.
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
 * @param output Unused parameter.
 */
template<typename T>
void PrintInputProcessing(const util::ParamData& d,
                          const void* input,
                          void* /* output */)
{
  PrintInputProcessing<T>(d, *((size_t*) input));
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
