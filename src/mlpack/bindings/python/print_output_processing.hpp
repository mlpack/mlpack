/**
 * @file print_output_processing.hpp
 * @author Ryan Curtin
 *
 * Print the output processing in a Python binding .pyx file for a given
 * parameter.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_OUTPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_OUTPUT_PROCESSING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Print output processing for a regular parameter type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * result = CLI.GetParam[int]('param_name')
     */
    std::cout << prefix << "result = CLI.GetParam[" << GetPythonType<T>(d)
        << "](\"" << d.name << "\")";
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['param_name'] = CLI.GetParam[int]('param_name')
     */
    std::cout << prefix << "result['" << d.name << "'] = CLI.GetParam["
        << GetPythonType<T>(d) << "]('" << d.name << "')" << std::endl;
  }
}

/**
 * Print output processing for a matrix type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * result = arma_numpy.mat_to_numpy_X(CLI.GetParam[mat]("name"))
     *
     * where X indicates the type to convert to.
     */
    std::cout << prefix << "result = arma_numpy.mat_to_numpy_"
        << GetNumpyTypeChar<T>() << "(CLI.GetParam[" << GetPythonType<T>(d)
        << "](\"" << d.name << "\"))" << std::endl;
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['param_name'] =
     *     arma_numpy.mat_to_numpy_X(CLI.GetParam[mat]('name')
     *
     * where X indicates the type to convert to.
     */
    std::cout << prefix << "result['" << d.name
        << "'] = arma_numpy.mat_to_numpy_" << GetNumpyTypeChar<T>()
        << "(CLI.GetParam[" << GetPythonType<T>(d) << "]('" << d.name << "'))"
        << std::endl;
  }
}

/**
 * Print output processing for a dataset info / matrix combination.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // Print the output with the matrix type.
  // We don't have any output of this type, so this is not implemented yet.
  // Let's just make something that will throw an error.
  std::cout << "invalid Python! see PrintOutputProcessing()!" << std::endl;
}

/**
 * Print output processing for a serializable model.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // Get the type names we need to use.
  std::string strippedType, printedType, defaultsType;
  StripType(d.cppType, strippedType, printedType, defaultsType);

  const std::string prefix(indent, ' ');

  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * result = ModelType()
     * MoveToPtr[Model]((<ModelType?> model).modelptr),
     *     CLI.GetParam[Model]('name'))
     */
    std::cout << prefix << "result = " << strippedType << "Type()" << std::endl;
    std::cout << prefix << "MoveToPtr[" << strippedType << "]((<"
        << strippedType << "Type?> result).modelptr, CLI.GetParam["
        << strippedType << "]('" << d.name << "'))" << std::endl;
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['name'] = ModelType()
     * MoveToPtr[Model*]((<ModelType?> result['name']).modelptr),
     *     CLI.GetParam[Model]('name'))
     */
    std::cout << prefix << "result['" << d.name << "'] = " << strippedType
        << "Type()" << std::endl;
    std::cout << prefix << "MoveToPtr[" << strippedType << "]((<"
        << strippedType << "Type?>" << " result['" << d.name
        << "']).modelptr, CLI.GetParam[" << strippedType << "]('"
        << d.name << "'))" << std::endl;
  }
}

/**
 * Given parameter information and the current number of spaces for indentation,
 * print the code to process the output to cout.  This code assumes that
 * data.input is false, and should not be called when data.input is true.  If
 * this is the only output, the results will be different.
 *
 * The input pointer should be a pointer to a std::tuple<size_t, bool> where the
 * first element is the indentation and the second element is a boolean
 * representing whether or not this is the only output parameter.
 *
 * @param d Parameter data struct.
 * @param input Pointer to size_t holding the indentation.
 * @param output Unused parameter.
 */
template<typename T>
void PrintOutputProcessing(const util::ParamData& d,
                           const void* input,
                           void* /* output */)
{
  std::tuple<size_t, bool>* tuple = (std::tuple<size_t, bool>*) input;

  PrintOutputProcessing<T>(d, std::get<0>(*tuple), std::get<1>(*tuple));
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
