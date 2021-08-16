/**
 * @file bindings/python/print_output_processing.hpp
 * @author Ryan Curtin
 *
 * Print the output processing in a Python binding .pyx file for a given
 * parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_OUTPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_OUTPUT_PROCESSING_HPP

#include <mlpack/prereqs.hpp>
#include "get_arma_type.hpp"
#include "get_numpy_type_char.hpp"
#include "get_cython_type.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Print output processing for a regular parameter type.
 */
template<typename T>
void PrintOutputProcessing(
    util::Params& /* params */,
    util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  const std::string prefix(indent, ' ');

  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * result = p.Get[int]('param_name')
     */
    std::cout << prefix << "result = " << "p.Get[" << GetCythonType<T>(d)
        << "](\"" << d.name << "\")";
    if (GetCythonType<T>(d) == "string")
    {
      std::cout << std::endl << prefix << "result = result.decode(\"UTF-8\")";
    }
    else if (GetCythonType<T>(d) == "vector[string]")
    {
      std::cout << std::endl << prefix
          << "result = [x.decode(\"UTF-8\") for x in result]";
    }
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['param_name'] = p.Get[int]('param_name')
     */
    std::cout << prefix << "result['" << d.name << "'] = p.Get["
        << GetCythonType<T>(d) << "](\"" << d.name << "\")" << std::endl;
    if (GetCythonType<T>(d) == "string")
    {
      std::cout << prefix << "result['" << d.name << "'] = result['" << d.name
          << "'].decode(\"UTF-8\")" << std::endl;
    }
    else if (GetCythonType<T>(d) == "vector[string]")
    {
      std::cout << prefix << "result['" << d.name << "'] = [x.decode(\"UTF-8\")"
          << " for x in result['" << d.name << "']]" << std::endl;
    }
  }
}

/**
 * Print output processing for a matrix type.
 */
template<typename T>
void PrintOutputProcessing(
    util::Params& /* params */,
    util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  const std::string prefix(indent, ' ');

  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * result = arma_numpy.mat_to_numpy_X(p.Get[mat]("name"))
     *
     * where X indicates the type to convert to.
     */
    std::cout << prefix << "result = arma_numpy." << GetArmaType<T>()
        << "_to_numpy_" << GetNumpyTypeChar<T>() << "(p.Get["
        << GetCythonType<T>(d) << "](\"" << d.name << "\"))" << std::endl;
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['param_name'] =
     *     arma_numpy.mat_to_numpy_X(p.Get[mat]('name')
     *
     * where X indicates the type to convert to.
     */
    std::cout << prefix << "result['" << d.name
        << "'] = arma_numpy." << GetArmaType<T>() << "_to_numpy_"
        << GetNumpyTypeChar<T>() << "(p.Get[" << GetCythonType<T>(d)
        << "]('" << d.name << "'))" << std::endl;
  }
}

/**
 * Print output processing for a dataset info / matrix combination.
 */
template<typename T>
void PrintOutputProcessing(
    util::Params& /* params */,
    util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  const std::string prefix(indent, ' ');

  // Print the output with the matrix type.  The dimension information doesn't
  // need to go back.
  if (onlyOutput)
  {
    /**
     * This gives us code like:
     *
     * result = arma_numpy.mat_to_numpy_X(GetParamWithInfo[mat](p, 'name'))
     */
    std::cout << prefix << "result = arma_numpy.mat_to_numpy_"
        << GetNumpyTypeChar<arma::mat>()
        << "(GetParamWithInfo[arma.Mat[double]](p, '" << d.name << "'))"
        << std::endl;
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['param_name'] =
     *     arma_numpy.mat_to_numpy_X(GetParamWithInfo[mat](p, 'name'))
     */
    std::cout << prefix << "result['" << d.name
        << "'] = arma_numpy.mat_to_numpy_" << GetNumpyTypeChar<arma::mat>()
        << "(GetParamWithInfo[arma.Mat[double]](p, '" << d.name << "'))"
        << std::endl;
  }
}

/**
 * Print output processing for a serializable model.
 */
template<typename T>
void PrintOutputProcessing(
    util::Params& params,
    util::ParamData& d,
    const size_t indent,
    const bool onlyOutput,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
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
     * (<ModelType?> result).modelptr = GetParamPtr[Model](p, 'name')
     */
    std::cout << prefix << "result = " << strippedType << "Type()" << std::endl;
    std::cout << prefix << "(<" << strippedType << "Type?> result).modelptr = "
        << "GetParamPtr[" << strippedType << "](p, '" << d.name << "')"
        << std::endl;

    /**
     * But we also have to check to ensure there aren't any input model
     * parameters of the same type that could have the same model pointer.
     * So we need to loop through all input parameters that have the same type,
     * and double-check.
     */
    std::map<std::string, util::ParamData>& parameters = params.Parameters();
    for (auto it = parameters.begin(); it != parameters.end(); ++it)
    {
      // Is it an input parameter of the same type?
      util::ParamData& data = it->second;
      if (data.input && data.cppType == d.cppType && data.required)
      {
        std::cout << prefix << "if (<" << strippedType
            << "Type> result).modelptr" << d.name << " == (<" << strippedType
            << "Type> " << data.name << ").modelptr:" << std::endl;
        std::cout << prefix << "  (<" << strippedType
            << "Type> result).modelptr = <" << strippedType << "*> 0"
            << std::endl;
        std::cout << prefix << "  result = " << data.name << std::endl;
      }
      else if (data.input && data.cppType == d.cppType)
      {
        std::cout << prefix << "if " << data.name << " is not None:"
            << std::endl;
        std::cout << prefix << "  if (<" << strippedType
            << "Type> result).modelptr" << d.name << " == (<" << strippedType
            << "Type> " << data.name << ").modelptr:" << std::endl;
        std::cout << prefix << "    (<" << strippedType
            << "Type> result).modelptr = <" << strippedType << "*> 0"
            << std::endl;
        std::cout << prefix << "    result = " << data.name << std::endl;
      }
    }
  }
  else
  {
    /**
     * This gives us code like:
     *
     * result['name'] = ModelType()
     * (<ModelType?> result['name']).modelptr = GetParamPtr[Model](p, 'name'))
     */
    std::cout << prefix << "result['" << d.name << "'] = " << strippedType
        << "Type()" << std::endl;
    std::cout << prefix << "(<" << strippedType << "Type?> result['" << d.name
        << "']).modelptr = GetParamPtr[" << strippedType << "](p, '" << d.name
        << "')" << std::endl;

    /**
     * But we also have to check to ensure there aren't any input model
     * parameters of the same type that could have the same model pointer.
     * So we need to loop through all input parameters that have the same type,
     * and double-check.
     */
    std::map<std::string, util::ParamData>& parameters = params.Parameters();
    for (auto it = parameters.begin(); it != parameters.end(); ++it)
    {
      // Is it an input parameter of the same type?
      util::ParamData& data = it->second;
      if (data.input && data.cppType == d.cppType && data.required)
      {
        std::cout << prefix << "if (<" << strippedType << "Type> result['"
            << d.name << "']).modelptr == (<" << strippedType << "Type> "
            << data.name << ").modelptr:" << std::endl;
        std::cout << prefix << "  (<" << strippedType << "Type> result['"
            << d.name << "']).modelptr = <" << strippedType << "*> 0"
            << std::endl;
        std::cout << prefix << "  result['" << d.name << "'] = " << data.name
            << std::endl;
      }
      else if (data.input && data.cppType == d.cppType)
      {
        std::cout << prefix << "if " << data.name << " is not None:"
            << std::endl;
        std::cout << prefix << "  if (<" << strippedType << "Type> result['"
            << d.name << "']).modelptr == (<" << strippedType << "Type> "
            << data.name << ").modelptr:" << std::endl;
        std::cout << prefix << "    (<" << strippedType << "Type> result['"
            << d.name << "']).modelptr = <" << strippedType << "*> 0"
            << std::endl;
        std::cout << prefix << "    result['" << d.name << "'] = " << data.name
            << std::endl;
      }
    }
  }
}

/**
 * Given parameter information and the current number of spaces for indentation,
 * print the code to process the output to cout.  This code assumes that
 * data.input is false, and should not be called when data.input is true.  If
 * this is the only output, the results will be different.
 *
 * The input pointer should be a pointer to a 
 * std::tuple<util::Params, std::tuple<size_t, bool>> where the first element is
 * the parameters of the binding and the second element is a tuple where the
 * first element is the indentation and the second element is a boolean
 * representing whether or not this is the only output parameter.
 *
 * @param d Parameter data struct.
 * @param input Pointer to size_t holding the indentation.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintOutputProcessing(util::ParamData& d,
                           const void* input,
                           void* /* output */)
{
  typedef std::tuple<util::Params, std::tuple<size_t, bool>> TupleType;
  TupleType* tuple = (TupleType*) input;

  PrintOutputProcessing<typename std::remove_pointer<T>::type>(std::get<0>(*tuple),
      d, std::get<0>(std::get<1>(*tuple)), std::get<1>(std::get<1>(*tuple)));
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
