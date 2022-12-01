/**
 * @file bindings/python/print_input_processing.hpp
 * @author Ryan Curtin
 * @author Yashwant Singh
 *
 * Print input processing for a Python binding option.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_INPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_INPUT_PROCESSING_HPP

#include <mlpack/prereqs.hpp>
#include "get_arma_type.hpp"
#include "get_numpy_type.hpp"
#include "get_numpy_type_char.hpp"
#include "get_cython_type.hpp"
#include "strip_type.hpp"
#include "wrapper_functions.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Print input processing for a standard option type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  // The copy_all_inputs parameter must be handled first, and therefore is
  // outside the scope of this code.
  if (d.name == "copy_all_inputs")
    return;

  const std::string prefix(indent, ' ');

  std::string def = "None";
  if (std::is_same<T, bool>::value)
    def = "False";

  // Make sure that we don't use names that are Python keywords.
  std::string name = GetValidName(d.name);

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   if isinstance(param_name, int):
   *     SetParam[int](p, \<const string\> 'param_name', param_name)
   *     p.SetPassed(\<const string\> 'param_name')
   *   else:
   *     raise TypeError("'param_name' must have type 'list'!")
   */
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;
  if (!d.required)
  {
    if (GetPrintableType<T>(d) == "bool")
    {
      std::cout << prefix << "if isinstance(" << name << ", "
          << GetPrintableType<T>(d) << "):" << std::endl;
      std::cout << prefix << "  if " << name << " is not " << def << ":"
          << std::endl;
    }
    else
    {
      std::cout << prefix << "if " << name << " is not " << def << ":"
          << std::endl;
      std::cout << prefix << "  if isinstance(" << name << ", "
          << GetPrintableType<T>(d) << "):" << std::endl;
    }

    std::cout << prefix << "    SetParam[" << GetCythonType<T>(d)
        << "](p, <const string> '" << d.name << "', ";
    if (GetCythonType<T>(d) == "string")
      std::cout << name << ".encode(\"UTF-8\")";
    else
      std::cout << name;
    std::cout << ")" << std::endl;
    std::cout << prefix << "    p.SetPassed(<const string> '" << d.name
        << "')" << std::endl;

    // If this parameter is "verbose", then enable verbose output.
    if (d.name == "verbose")
    std::cout << prefix << "    EnableVerbose()" << std::endl;

    if (GetPrintableType<T>(d) == "bool")
    {
      std::cout << "  else:" << std::endl;
      std::cout << "    raise TypeError(" <<"\"'"<< name
          << "' must have type \'" << GetPrintableType<T>(d)
          << "'!\")" << std::endl;
    }
    else
    {
      std::cout << "    else:" << std::endl;
      std::cout << "      raise TypeError(" <<"\"'"<< name
          << "' must have type \'" << GetPrintableType<T>(d)
          << "'!\")" << std::endl;
    }
  }
  else
  {
    if (GetPrintableType<T>(d) == "bool")
    {
      std::cout << prefix << "if isinstance(" << name << ", "
          << GetPrintableType<T>(d) << "):" << std::endl;
      std::cout << prefix << "  if " << name << " is not " << def << ":"
          << std::endl;
    }
    else
    {
      std::cout << prefix << "if " << name << " is not " << def << ":"
          << std::endl;
      std::cout << prefix << "  if isinstance(" << name << ", "
          << GetPrintableType<T>(d) << "):" << std::endl;
    }

    std::cout << prefix << "    SetParam[" << GetCythonType<T>(d) << "](p, <const "
        << "string> '" << d.name << "', ";
    if (GetCythonType<T>(d) == "string")
      std::cout << name << ".encode(\"UTF-8\")";
    else if (GetCythonType<T>(d) == "vector[string]")
      std::cout << "[i.encode(\"UTF-8\") for i in " << name << "]";
    else
      std::cout << name;
    std::cout << ")" << std::endl;
    std::cout << prefix << "    p.SetPassed(<const string> '"
        << d.name << "')" << std::endl;

    if (GetPrintableType<T>(d) == "bool")
    {
      std::cout << "  else:" << std::endl;
      std::cout << "    raise TypeError(" <<"\"'"<< name
          << "' must have type \'" << GetPrintableType<T>(d)
          << "'!\")" << std::endl;
    }
    else
    {
      std::cout << "    else:" << std::endl;
      std::cout << "      raise TypeError(" <<"\"'"<< name
          << "' must have type \'" << GetPrintableType<T>(d)
          << "'!\")" << std::endl;
    }
  }
  std::cout << std::endl; // Extra line is to clear up the code a bit.
}

/**
 * Print input processing for a vector type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *  if param_name is not None:
   *    if isinstance(param_name, list):
   *      if len(param_name) > 0:
   *        if isinstance(param_name[0], str):
   *          SetParam[vector[string]](p, \<const string\> 'param_name', param_name)
   *          p.SetPassed(\<const string\> 'param_name')
   *        else:
   *          raise TypeError("'param_name' must have type 'list of strs'!")
   *    else:
   *      raise TypeError("'param_name' must have type 'list'!")
   *
   */
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;

  std::string name = GetValidName(d.name);

  if (!d.required)
  {
    std::cout << prefix << "if " << name << " is not None:"
        << std::endl;
    std::cout << prefix << "  if isinstance(" << name << ", list):"
        << std::endl;
    std::cout << prefix << "    if len(" << name << ") > 0:"
        << std::endl;
    std::cout << prefix << "      if isinstance(" << name << "[0], "
        << GetPrintableType<typename T::value_type>(d) << "):" << std::endl;
    std::cout << prefix << "        SetParam[" << GetCythonType<T>(d)
        << "](p, <const string> '" << d.name << "', ";
    // Strings need special handling.
    if (GetCythonType<T>(d) == "vector[string]")
      std::cout << "[i.encode(\"UTF-8\") for i in " << name << "]";
    else
      std::cout << name;
    std::cout << ")" << std::endl;
    std::cout << prefix << "        p.SetPassed(<const string> '" << d.name
        << "')" << std::endl;
    std::cout << prefix << "      else:" << std::endl;
    std::cout << prefix << "        raise TypeError(" <<"\"'"<< d.name
        << "' must have type \'" << GetPrintableType<T>(d)
        << "'!\")" << std::endl;
    std::cout << prefix << "  else:" << std::endl;
    std::cout << prefix << "    raise TypeError(" <<"\"'"<< d.name
        << "' must have type \'list'!\")" << std::endl;
  }
  else
  {
    std::cout << prefix << "if isinstance(" << name << ", list):"
        << std::endl;
    std::cout << prefix << "  if len(" << name << ") > 0:"
        << std::endl;
    std::cout << prefix << "    if isinstance(" << name << "[0], "
        << GetPrintableType<typename T::value_type>(d) << "):" << std::endl;
    std::cout << prefix << "      SetParam[" << GetCythonType<T>(d)
        << "](p, <const string> '" << d.name << "', ";
    // Strings need special handling.
    if (GetCythonType<T>(d) == "vector[string]")
      std::cout << "[i.encode(\"UTF-8\") for i in " << name << "]";
    else
      std::cout << name;
    std::cout << ")" << std::endl;
    std::cout << prefix << "      p.SetPassed(<const string> '" << d.name
        << "')" << std::endl;
    std::cout << prefix << "    else:" << std::endl;
    std::cout << prefix << "      raise TypeError(" <<"\"'"<< d.name
        << "' must have type \'" << GetPrintableType<T>(d)
        << "'!\")" << std::endl;
    std::cout << prefix << "else:" << std::endl;
    std::cout << prefix << "  raise TypeError(" <<"\"'"<< d.name
        << "' must have type \'list'!\")" << std::endl;
  }
}

/**
 * Print input processing for a matrix type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   param_name_tuple = to_matrix(param_name)
   *   if param_name_tuple[0].shape[0] == 1 or
   *       param_name_tuple[0].shape[1] == 1:
   *     param_name_tuple[0].shape = (param_name_tuple[0].size,)
   *   param_name_mat = arma_numpy.numpy_to_mat_s(param_name_tuple[0],
   *       param_name_tuple[1])
   *   SetParam[mat](p, \<const string\> 'param_name', dereference(param_name_mat), True)
   *   p.SetPassed(\<const string\> 'param_name')
   *
   * The value of the final boolean passed to SetParam is determined by whether
   * the matrix is transposed or not.  That boolean is omitted if the parameter
   * is a row or column.
   */
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;
  std::string name = GetValidName(d.name);
  std::string transStr =
      (d.noTranspose ? std::string("True") : std::string("False"));

  if (!d.required)
  {
    if (T::is_row || T::is_col)
    {
      std::cout << prefix << "if " << name << " is not None:" << std::endl;
      std::cout << prefix << "  " << name << "_tuple = to_matrix("
          << name << ", dtype=" << GetNumpyType<typename T::elem_type>()
          << ", copy=p.Has('copy_all_inputs'))" << std::endl;
      std::cout << prefix << "  if len(" << name << "_tuple[0].shape) > 1:"
          << std::endl;
      std::cout << prefix << "    if " << name << "_tuple[0]"
          << ".shape[0] == 1 or " << name << "_tuple[0].shape[1] == 1:"
          << std::endl;
      std::cout << prefix << "      " << name << "_tuple[0].shape = ("
          << d.name << "_tuple[0].size,)" << std::endl;
      std::cout << prefix << "  " << name << "_mat = arma_numpy.numpy_to_"
          << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << name
          << "_tuple[0], " << name << "_tuple[1])" << std::endl;
      std::cout << prefix << "  SetParam[" << GetCythonType<T>(d)
          << "](p, <const string> '" << d.name << "', dereference("
          << name << "_mat))" << std::endl;
      std::cout << prefix << "  p.SetPassed(<const string> '" << d.name
          << "')" << std::endl;
      std::cout << prefix << "  del " << name << "_mat" << std::endl;
    }
    else
    {
      std::cout << prefix << "if " << name << " is not None:" << std::endl;
      std::cout << prefix << "  " << name << "_tuple = to_matrix("
          << name << ", dtype=" << GetNumpyType<typename T::elem_type>()
          << ", copy=p.Has('copy_all_inputs'))" << std::endl;
      std::cout << prefix << "  if len(" << name << "_tuple[0].shape"
          << ") < 2:" << std::endl;
      std::cout << prefix << "    " << name << "_tuple[0].shape = (" << name
          << "_tuple[0].shape[0], 1)" << std::endl;
      std::cout << prefix << "  " << name << "_mat = arma_numpy.numpy_to_"
          << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << name
          << "_tuple[0], " << name << "_tuple[1])" << std::endl;
      std::cout << prefix << "  SetParam[" << GetCythonType<T>(d)
          << "](p, <const string> '" << d.name << "', dereference("
          << name << "_mat), " << transStr << ")" << std::endl;
      std::cout << prefix << "  p.SetPassed(<const string> '" << d.name
          << "')" << std::endl;
      std::cout << prefix << "  del " << name << "_mat" << std::endl;
    }
  }
  else
  {
    if (T::is_row || T::is_col)
    {
      std::cout << prefix << name << "_tuple = to_matrix(" << name
          << ", dtype=" << GetNumpyType<typename T::elem_type>()
          << ", copy=p.Has('copy_all_inputs'))" << std::endl;
      std::cout << prefix << "if len(" << name << "_tuple[0].shape) > 1:"
          << std::endl;
      std::cout << prefix << "  if " << name << "_tuple[0].shape[0] == 1 or "
          << name << "_tuple[0].shape[1] == 1:" << std::endl;
      std::cout << prefix << "    " << name << "_tuple[0].shape = ("
          << name << "_tuple[0].size,)" << std::endl;
      std::cout << prefix << name << "_mat = arma_numpy.numpy_to_"
          << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << name
          << "_tuple[0], " << name << "_tuple[1])" << std::endl;
      std::cout << prefix << "SetParam[" << GetCythonType<T>(d)
          << "](p, <const string> '" << d.name << "', dereference("
          << name << "_mat))" << std::endl;
      std::cout << prefix << "p.SetPassed(<const string> '" << d.name << "')"
          << std::endl;
      std::cout << prefix << "del " << name << "_mat" << std::endl;
    }
    else
    {
      std::cout << prefix << name << "_tuple = to_matrix(" << name
          << ", dtype=" << GetNumpyType<typename T::elem_type>()
          << ", copy=p.Has('copy_all_inputs'))" << std::endl;
      std::cout << prefix << "if len(" << name << "_tuple[0].shape) < 2:"
          << std::endl;
      std::cout << prefix << "  " << name << "_tuple[0].shape = (" << name
          << "_tuple[0].shape[0], 1)" << std::endl;
      std::cout << prefix << name << "_mat = arma_numpy.numpy_to_"
          << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << name
          << "_tuple[0], " << name << "_tuple[1])" << std::endl;
      std::cout << prefix << "SetParam[" << GetCythonType<T>(d)
          << "](p, <const string> '" << d.name << "', dereference(" << name
          << "_mat), " << transStr << ")" << std::endl;
      std::cout << prefix << "p.SetPassed(<const string> '" << d.name << "')"
          << std::endl;
      std::cout << prefix << "del " << name << "_mat" << std::endl;
    }
  }
  std::cout << std::endl;
}

/**
 * Print input processing for a serializable type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // First, get the correct class name if needed.
  std::string strippedType, printedType, defaultsType;
  StripType(d.cppType, strippedType, printedType, defaultsType);

  std::string name = GetValidName(d.name);

  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   * # Detect if the parameter was passed; set if so.
   * if param_name is not None:
   *   try:
   *     SetParamPtr[Model](p, 'param_name', (\<ModelType?\> param_name).modelptr,
   *         p.Has('copy_all_inputs'))
   *   except TypeError as e:
   *     if type(param_name).__name__ == "ModelType":
   *       SetParamPtr[Model](p, 'param_name', (\<ModelType\> param_name).modelptr,
   *           p.Has('copy_all_inputs'))
   *     else:
   *       raise e
   *   p.SetPassed(<const string> 'param_name')
   */
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "if " << name << " is not None:" << std::endl;
    std::cout << prefix << "  try:" << std::endl;
    std::cout << prefix << "    SetParamPtr[" << strippedType << "](p, '" << d.name
        << "', (<" << strippedType << "Type?> " << name << ").modelptr, "
        << "p.Has('copy_all_inputs'))" << std::endl;
    std::cout << prefix << "  except TypeError as e:" << std::endl;
    std::cout << prefix << "    if type(" << name << ").__name__ == '"
        << strippedType << "Type':" << std::endl;
    std::cout << prefix << "      SetParamPtr[" << strippedType << "](p, '"
        << d.name << "', (<" << strippedType << "Type> " << name
        << ").modelptr, p.Has('copy_all_inputs'))" << std::endl;
    std::cout << prefix << "    else:" << std::endl;
    std::cout << prefix << "      raise e" << std::endl;
    std::cout << prefix << "  p.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
  }
  else
  {
    std::cout << prefix << "try:" << std::endl;
    std::cout << prefix << "  SetParamPtr[" << strippedType << "](p, '" << d.name
        << "', (<" << strippedType << "Type?> " << name << ").modelptr, "
        << "p.Has('copy_all_inputs'))" << std::endl;
    std::cout << prefix << "except TypeError as e:" << std::endl;
    std::cout << prefix << "  if type(" << name << ").__name__ == '"
        << strippedType << "Type':" << std::endl;
    std::cout << prefix << "    SetParamPtr[" << strippedType << "](p,'" << d.name
        << "', (<" << strippedType << "Type> " << name << ").modelptr, "
        << "p.Has('copy_all_inputs'))" << std::endl;
    std::cout << prefix << "  else:" << std::endl;
    std::cout << prefix << "    raise e" << std::endl;
    std::cout << prefix << "p.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
  }
  std::cout << std::endl;
}

/**
 * Print input processing for a matrix/DatasetInfo type.
 */
template<typename T>
void PrintInputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  std::string name = GetValidName(d.name);

  // The user should pass in a matrix type of some sort.
  const std::string prefix(indent, ' ');

  /** We want to generate code like the following:
   *
   * cdef extern from "numpy/arrayobject.h":
   *   void* PyArray_DATA(np.ndarray arr)
   * if param_name is not None:
   *   param_name_tuple = to_matrix_with_info(param_name)
   *   if len(param_name_tuple[0].shape) < 2:
   *     param_name_tuple[0].shape = (param_name_tuple[0].size,)
   *   param_name_mat = arma_numpy.numpy_to_matrix_d(param_name_tuple[0])
   *   SetParamWithInfo[mat](p, \<const string\> 'param_name',
   *       dereference(param_name_mat), 
   *       \<const cbool*\> PyArray_DATA(param_name_dims))
   *   p.SetPassed(\<const string\> 'param_name')
   */
  std::cout << prefix << "cdef np.ndarray " << name << "_dims" << std::endl;
  std::cout << prefix << "# Detect if the parameter was passed; set if so."
      << std::endl;
  if (!d.required)
  {
    std::cout << prefix << "cdef extern from \"numpy/arrayobject.h\":" << std::endl;
    std::cout << prefix << "  void* PyArray_DATA(np.ndarray arr)" << std::endl;
    std::cout << prefix << "if " << d.name << " is not None:" << std::endl;
    std::cout << prefix << "  " << d.name << "_tuple = to_matrix_with_info("
        << d.name << ", dtype=np.double, copy=p.Has('copy_all_inputs'))"
        << std::endl;
    std::cout << prefix << "  if len(" << name << "_tuple[0].shape"
        << ") < 2:" << std::endl;
    std::cout << prefix << "    " << name << "_tuple[0].shape = (" << name
        << "_tuple[0].shape[0], 1)" << std::endl;
    std::cout << prefix << "  " << name << "_mat = arma_numpy.numpy_to_mat_d("
        << name << "_tuple[0], " << name << "_tuple[1])" << std::endl;
    std::cout << prefix << "  " << name << "_dims = " << name
        << "_tuple[2]" << std::endl;
    std::cout << prefix << "  SetParamWithInfo[arma.Mat[double]](p, <const "
        << "string> '" << d.name << "', dereference(" << d.name << "_mat), "
        << "<const cbool*> PyArray_DATA(" << d.name << "_dims))" << std::endl;
    std::cout << prefix << "  p.SetPassed(<const string> '" << d.name
        << "')" << std::endl;
    std::cout << prefix << "  del " << name << "_mat" << std::endl;
  }
  else
  {
    std::cout << prefix << "cdef extern from \"numpy/arrayobject.h\":" << std::endl;
    std::cout << prefix << "  void* PyArray_DATA(np.ndarray arr)" << std::endl;
    std::cout << prefix << d.name << "_tuple = to_matrix_with_info(" << d.name
        << ", dtype=np.double, copy=p.Has('copy_all_inputs'))"
        << std::endl;
    std::cout << prefix << "if len(" << name << "_tuple[0].shape"
        << ") < 2:" << std::endl;
    std::cout << prefix << "  " << name << "_tuple[0].shape = (" << name
        << "_tuple[0].shape[0], 1)" << std::endl;
    std::cout << prefix << name << "_mat = arma_numpy.numpy_to_mat_d("
        << name << "_tuple[0], " << name << "_tuple[1])" << std::endl;
    std::cout << prefix << name << "_dims = " << name << "_tuple[2]"
        << std::endl;
    std::cout << prefix << "SetParamWithInfo[arma.Mat[double]](p, <const "
        << "string> '" << d.name << "', dereference(" << d.name << "_mat), "
        << "<const cbool*> PyArray_DATA(" << d.name << "_dims))" << std::endl;
    std::cout << prefix << "p.SetPassed(<const string> '" << d.name << "')"
        << std::endl;
    std::cout << prefix << "del " << name << "_mat" << std::endl;
  }
  std::cout << std::endl;
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

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
