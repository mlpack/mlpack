/**
 * @file print_functions.cpp
 * @author Ryan Curtin
 *
 * Implementation of utility functions to print information about parameters.
 */
#include "print_functions.hpp"
#include "type_names.hpp"
#include <mlpack/core/util/hyphenate_string.hpp>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Given parameter info, print the definition of the parameter to cout.  You are
 * responsible for setting up the line---this does not handle indentation or
 * anything.
 */
void PrintDefinition(const ParamData& data)
{
  cout << data.name;
  if (!data.required && !data.isFlag)
    cout << "=None";
  else if (data.isFlag) // Get correct default type for flags.
    cout << "=False";
}

/**
 * Given an option, print documentation for it to cout.  You are responsible for
 * setting up the line---this does not handle indentation or anything.  This is
 * meant to produce a line of documentation for the docstring of the Python
 * function, describing a single parameter.
 *
 * The indent parameter should be passed to know how much to indent for a new
 * line.
 */
template<typename T>
void PrintDocumentation(
    const ParamData& data,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
{
  cout << data.name << " (" << Typename(data) << "): " <<
      HyphenateString(data.desc, indent + 4);
  cout << " Default value " << data.defaultFunction(data) << ".";
}

/**
 * If the type is not something we can print a default value for, print the
 * documentation but not the default value.
 */
template<typename T>
void PrintDocumentation(
    const ParamData& data,
    const size_t indent,
    const typename boost::enable_if_c<
        arma::is_arma_type<T>::value ||
        std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                   arma::mat>>::value ||
        data::HasSerialize<T>::value>::type* = 0)
{
  cout << data.name << " (" << Typename<T>() << "): " <<
      HyphenateString(data.desc, indent + 4);
}

/**
 * Given parameter info and the current number of spaces for indentation, print
 * the code to process the input to cout.  This code assumes that data.input is
 * true, and should not be called when data.input is false.
 *
 * @param data Parameter data.
 * @param indent Number of spaces to indent.
 */
template<typename T>
void PrintInputProcessing(
    const ParamData& data,
    const size_t indent,
    const typename boost::disable_if<is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   SetParam[int](<const string> 'param_name', param_name)
   */
  cout << prefix << "# Detect if the parameter was passed; set if so.";
  cout << endl;
  if (IsMatrixType(data))
  {
    cout << prefix << "cdef " << Typename(data) << "* " << data.name << "_mat"
        << endl;
  }

  if (!data.required)
  {
    cout << prefix << "if " << data.name << " is not None:" << endl;

    // If it's a matrix type, we have to convert it first.
    if (IsMatrixType(data))
    {
      cout << prefix << "  " << data.name << "_mat = arma_numpy.numpy_to_mat_"
          << MatrixTypeSuffix(data) << "(" << data.name << ")" << endl;
    }

    cout << prefix << "  SetParam[" << Typename(data) << "](<const string> '"
        << data.name << "', ";
    if (IsMatrixType(data))
      cout << "dereference(" << data.name << "_mat)";
    else
      cout << data.name;
    cout << ")" << endl;
    cout << prefix << "  CLI.SetPassed(<const string> '" << data.name << "')"
        << endl;
  }
  else
  {
    // If it's a matrix type, we have to convert it first.
    if (IsMatrixType(data))
    {
      cout << prefix << data.name << "_mat = arma_numpy.numpy_to_mat_"
          << MatrixTypeSuffix(data) << "(" << data.name << ")" << endl;
    }

    cout << prefix << "SetParam[" << Typename(data) << "](<const string> '"
        << data.name << "', ";
    if (IsMatrixType(data))
      cout << "dereference(" << data.name << "_mat)";
    else
      cout << data.name;
    cout << ")" << endl;
    cout << prefix << "CLI.SetPassed(<const string> '" << data.name << "')"
        << endl;
  }
  cout << endl; // Extra line is to clear up the code a bit.
}

template<typename T>
void PrintInputProcessing(
    const ParamData& data,
    const size_t indent,
    const typename boost::enable_if<is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   SetParam[int](<const string> 'param_name', param_name)
   */
  cout << prefix << "# Detect if the parameter was passed; set if so.";
  cout << endl;
  cout << prefix << "cdef " << Typename(data) << "* " << data.name << "_mat"
      << endl;

  if (!data.required)
  {
    cout << prefix << "if " << data.name << " is not None:" << endl;

    cout << prefix << "  " << data.name << "_mat = arma_numpy.numpy_to_mat_"
        << MatrixTypeSuffix(data) << "(" << data.name << ")" << endl;

    cout << prefix << "  SetParam[" << Typename(data) << "](<const string> '"
        << data.name << "', dereference(" << data.name << "_mat))" << endl;
    cout << prefix << "  CLI.SetPassed(<const string> '" << data.name << "')"
        << endl;
  }
  else
  {
    cout << prefix << data.name << "_mat = arma_numpy.numpy_to_mat_"
        << MatrixTypeSuffix(data) << "(" << data.name << ")" << endl;
    cout << prefix << "SetParam[" << Typename(data) << "](<const string> '"
        << data.name << "', dereference(" << data.name << "_mat))" << endl;
    cout << prefix << "CLI.SetPassed(<const string> '" << data.name << "')"
        << endl;
  }
  cout << endl; // Extra line is to clear up the code a bit.
}

/**
 * Given parameter info and the current number of spaces for indentation, print
 * the code to process the output to cout.  This code assumes that data.input is
 * false, and should not be called when data.input is true.
 *
 * @param data Parameter data.
 * @param indent Number of spaces to indent.
 * @param onlyOutput This should be set to true if this is the only output
 *      parameter.
 */
void PrintOutputProcessing(
    const ParamData& data,
    const size_t indent,
    const bool onlyOutput,
    const typename boost::disable_if<is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * return CLI.GetParam[int]('param_name')
     */
    cout << prefix << "return CLI.GetParam[" << Typename(data) << "](\""
        << data.name << "\")";
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['param_name'] = CLI.GetParam[int]('param_name')
     */
    cout << prefix << "result['" << data.name << "'] = CLI.GetParam["
        << Typename(data) << "]('" << data.name << "')" << endl;
  }
}

void PrintOutputProcessing(
    const ParamData& data,
    const size_t indent,
    const bool onlyOutput,
    const typename boost::enable_if<is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * return CLI.GetParam[int]('param_name')
     */
    cout << prefix << "return arma_numpy.mat_to_numpy_"
        << MatrixTypeSuffix(data) << "(CLI.GetParam[" << Typename(data)
        << "](\"" << data.name << "\"))" << endl;
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['param_name'] = CLI.GetParam[int]('param_name')
     */
    cout << prefix << "result['" << data.name << "'] = arma_numpy.mat_to_numpy_"
        << MatrixTypeSuffix(data) << "(CLI.GetParam[" << Typename(data)
        << "]('" << data.name << "'))" << endl;
  }
}

template<typename T>
void PrintClassDefinition(
    const ParamData& data,
    const size_t indent,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
{
  // Don't print anything.
}

template<typename T>
void PrintClassDefinition(
    const ParamData& data,
    const size_t indent,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  /**
   * This will produce code like:
   *
   * cdef class <ModelType>Type:
   *   cdef <ModelType>* modelptr
   *
   *   def __init__(self):
   *     self.modelptr = new <ModelType>()
   *
   *   def __dealloc__(self):
   *     del self.modelptr
   */
  std::cout << "cdef class " << d.cppType << "Type:" << std::endl;
  std::cout << "  cdef " << d.cppType << "* modelptr" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __init__(self):" << std::endl;
  std::cout << "    self.modelptr = new " << d.cppType << "()" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __dealloc__(self):" << std::endl;
  std::cout << "    del self.modelptr" << std::endl;
  std::cout << std::endl;

}

} // namespace python
} // namespace bindings
} // namespace mlpack
