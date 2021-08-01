
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_print_input_processing.hpp:

Program Listing for File print_input_processing.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_print_input_processing.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/print_input_processing.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_PRINT_INPUT_PROCESSING_HPP
   #define MLPACK_BINDINGS_PYTHON_PRINT_INPUT_PROCESSING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "get_arma_type.hpp"
   #include "get_numpy_type.hpp"
   #include "get_numpy_type_char.hpp"
   #include "get_cython_type.hpp"
   #include "strip_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
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
     std::string name = (d.name == "lambda") ? "lambda_" : d.name;
   
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
           << "](<const string> '" << d.name << "', ";
       if (GetCythonType<T>(d) == "string")
         std::cout << name << ".encode(\"UTF-8\")";
       else
         std::cout << name;
       std::cout << ")" << std::endl;
       std::cout << prefix << "    IO.SetPassed(<const string> '" << d.name
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
   
       std::cout << prefix << "    SetParam[" << GetCythonType<T>(d) << "](<const "
           << "string> '" << d.name << "', ";
       if (GetCythonType<T>(d) == "string")
         std::cout << name << ".encode(\"UTF-8\")";
       else if (GetCythonType<T>(d) == "vector[string]")
         std::cout << "[i.encode(\"UTF-8\") for i in " << name << "]";
       else
         std::cout << name;
       std::cout << ")" << std::endl;
       std::cout << prefix << "    IO.SetPassed(<const string> '"
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
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0,
       const typename boost::enable_if<util::IsStdVector<T>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::cout << prefix << "# Detect if the parameter was passed; set if so."
         << std::endl;
     if (!d.required)
     {
       std::cout << prefix << "if " << d.name << " is not None:"
           << std::endl;
       std::cout << prefix << "  if isinstance(" << d.name << ", list):"
           << std::endl;
       std::cout << prefix << "    if len(" << d.name << ") > 0:"
           << std::endl;
       std::cout << prefix << "      if isinstance(" << d.name << "[0], "
           << GetPrintableType<typename T::value_type>(d) << "):" << std::endl;
       std::cout << prefix << "        SetParam[" << GetCythonType<T>(d)
           << "](<const string> '" << d.name << "', ";
       // Strings need special handling.
       if (GetCythonType<T>(d) == "vector[string]")
         std::cout << "[i.encode(\"UTF-8\") for i in " << d.name << "]";
       else
         std::cout << d.name;
       std::cout << ")" << std::endl;
       std::cout << prefix << "        IO.SetPassed(<const string> '" << d.name
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
       std::cout << prefix << "if isinstance(" << d.name << ", list):"
           << std::endl;
       std::cout << prefix << "  if len(" << d.name << ") > 0:"
           << std::endl;
       std::cout << prefix << "    if isinstance(" << d.name << "[0], "
           << GetPrintableType<typename T::value_type>(d) << "):" << std::endl;
       std::cout << prefix << "      SetParam[" << GetCythonType<T>(d)
           << "](<const string> '" << d.name << "', ";
       // Strings need special handling.
       if (GetCythonType<T>(d) == "vector[string]")
         std::cout << "[i.encode(\"UTF-8\") for i in " << d.name << "]";
       else
         std::cout << d.name;
       std::cout << ")" << std::endl;
       std::cout << prefix << "      IO.SetPassed(<const string> '" << d.name
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
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::cout << prefix << "# Detect if the parameter was passed; set if so."
         << std::endl;
     if (!d.required)
     {
       if (T::is_row || T::is_col)
       {
         std::cout << prefix << "if " << d.name << " is not None:" << std::endl;
         std::cout << prefix << "  " << d.name << "_tuple = to_matrix("
             << d.name << ", dtype=" << GetNumpyType<typename T::elem_type>()
             << ", copy=IO.HasParam('copy_all_inputs'))" << std::endl;
         std::cout << prefix << "  if len(" << d.name << "_tuple[0].shape) > 1:"
             << std::endl;
         std::cout << prefix << "    if " << d.name << "_tuple[0]"
             << ".shape[0] == 1 or " << d.name << "_tuple[0].shape[1] == 1:"
             << std::endl;
         std::cout << prefix << "      " << d.name << "_tuple[0].shape = ("
             << d.name << "_tuple[0].size,)" << std::endl;
         std::cout << prefix << "  " << d.name << "_mat = arma_numpy.numpy_to_"
             << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << d.name
             << "_tuple[0], " << d.name << "_tuple[1])" << std::endl;
         std::cout << prefix << "  SetParam[" << GetCythonType<T>(d)
             << "](<const string> '" << d.name << "', dereference("
             << d.name << "_mat))"<< std::endl;
         std::cout << prefix << "  IO.SetPassed(<const string> '" << d.name
             << "')" << std::endl;
         std::cout << prefix << "  del " << d.name << "_mat" << std::endl;
       }
       else
       {
         std::cout << prefix << "if " << d.name << " is not None:" << std::endl;
         std::cout << prefix << "  " << d.name << "_tuple = to_matrix("
             << d.name << ", dtype=" << GetNumpyType<typename T::elem_type>()
             << ", copy=IO.HasParam('copy_all_inputs'))" << std::endl;
         std::cout << prefix << "  if len(" << d.name << "_tuple[0].shape"
             << ") < 2:" << std::endl;
         std::cout << prefix << "    " << d.name << "_tuple[0].shape = (" << d.name
             << "_tuple[0].shape[0], 1)" << std::endl;
         std::cout << prefix << "  " << d.name << "_mat = arma_numpy.numpy_to_"
             << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << d.name
             << "_tuple[0], " << d.name << "_tuple[1])" << std::endl;
         std::cout << prefix << "  SetParam[" << GetCythonType<T>(d)
             << "](<const string> '" << d.name << "', dereference("
             << d.name << "_mat))"<< std::endl;
         std::cout << prefix << "  IO.SetPassed(<const string> '" << d.name
             << "')" << std::endl;
         std::cout << prefix << "  del " << d.name << "_mat" << std::endl;
       }
     }
     else
     {
       if (T::is_row || T::is_col)
       {
         std::cout << prefix << d.name << "_tuple = to_matrix(" << d.name
             << ", dtype=" << GetNumpyType<typename T::elem_type>()
             << ", copy=IO.HasParam('copy_all_inputs'))" << std::endl;
         std::cout << prefix << "if len(" << d.name << "_tuple[0].shape) > 1:"
             << std::endl;
         std::cout << prefix << "  if " << d.name << "_tuple[0].shape[0] == 1 or "
             << d.name << "_tuple[0].shape[1] == 1:" << std::endl;
         std::cout << prefix << "    " << d.name << "_tuple[0].shape = ("
             << d.name << "_tuple[0].size,)" << std::endl;
         std::cout << prefix << d.name << "_mat = arma_numpy.numpy_to_"
             << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << d.name
             << "_tuple[0], " << d.name << "_tuple[1])" << std::endl;
         std::cout << prefix << "SetParam[" << GetCythonType<T>(d)
             << "](<const string> '" << d.name << "', dereference("
             << d.name << "_mat))"<< std::endl;
         std::cout << prefix << "IO.SetPassed(<const string> '" << d.name << "')"
             << std::endl;
         std::cout << prefix << "del " << d.name << "_mat" << std::endl;
       }
       else
       {
         std::cout << prefix << d.name << "_tuple = to_matrix(" << d.name
             << ", dtype=" << GetNumpyType<typename T::elem_type>()
             << ", copy=IO.HasParam('copy_all_inputs'))" << std::endl;
         std::cout << prefix << "if len(" << d.name << "_tuple[0].shape) > 2:"
             << std::endl;
         std::cout << prefix << "  " << d.name << "_tuple[0].shape = (" << d.name
             << "_tuple[0].shape[0], 1)" << std::endl;
         std::cout << prefix << d.name << "_mat = arma_numpy.numpy_to_"
             << GetArmaType<T>() << "_" << GetNumpyTypeChar<T>() << "(" << d.name
             << "_tuple[0], " << d.name << "_tuple[1])" << std::endl;
         std::cout << prefix << "SetParam[" << GetCythonType<T>(d)
             << "](<const string> '" << d.name << "', dereference(" << d.name
             << "_mat))" << std::endl;
         std::cout << prefix << "IO.SetPassed(<const string> '" << d.name << "')"
             << std::endl;
         std::cout << prefix << "del " << d.name << "_mat" << std::endl;
       }
     }
     std::cout << std::endl;
   }
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // First, get the correct class name if needed.
     std::string strippedType, printedType, defaultsType;
     StripType(d.cppType, strippedType, printedType, defaultsType);
   
     const std::string prefix(indent, ' ');
   
     std::cout << prefix << "# Detect if the parameter was passed; set if so."
         << std::endl;
     if (!d.required)
     {
       std::cout << prefix << "if " << d.name << " is not None:" << std::endl;
       std::cout << prefix << "  try:" << std::endl;
       std::cout << prefix << "    SetParamPtr[" << strippedType << "]('" << d.name
           << "', (<" << strippedType << "Type?> " << d.name << ").modelptr, "
           << "IO.HasParam('copy_all_inputs'))" << std::endl;
       std::cout << prefix << "  except TypeError as e:" << std::endl;
       std::cout << prefix << "    if type(" << d.name << ").__name__ == '"
           << strippedType << "Type':" << std::endl;
       std::cout << prefix << "      SetParamPtr[" << strippedType << "]('"
           << d.name << "', (<" << strippedType << "Type> " << d.name
           << ").modelptr, IO.HasParam('copy_all_inputs'))" << std::endl;
       std::cout << prefix << "    else:" << std::endl;
       std::cout << prefix << "      raise e" << std::endl;
       std::cout << prefix << "  IO.SetPassed(<const string> '" << d.name << "')"
           << std::endl;
     }
     else
     {
       std::cout << prefix << "try:" << std::endl;
       std::cout << prefix << "  SetParamPtr[" << strippedType << "]('" << d.name
           << "', (<" << strippedType << "Type?> " << d.name << ").modelptr, "
           << "IO.HasParam('copy_all_inputs'))" << std::endl;
       std::cout << prefix << "except TypeError as e:" << std::endl;
       std::cout << prefix << "  if type(" << d.name << ").__name__ == '"
           << strippedType << "Type':" << std::endl;
       std::cout << prefix << "    SetParamPtr[" << strippedType << "]('" << d.name
           << "', (<" << strippedType << "Type> " << d.name << ").modelptr, "
           << "IO.HasParam('copy_all_inputs'))" << std::endl;
       std::cout << prefix << "  else:" << std::endl;
       std::cout << prefix << "    raise e" << std::endl;
       std::cout << prefix << "IO.SetPassed(<const string> '" << d.name << "')"
           << std::endl;
     }
     std::cout << std::endl;
   }
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     // The user should pass in a matrix type of some sort.
     const std::string prefix(indent, ' ');
   
     std::cout << prefix << "cdef np.ndarray " << d.name << "_dims" << std::endl;
     std::cout << prefix << "# Detect if the parameter was passed; set if so."
         << std::endl;
     if (!d.required)
     {
       std::cout << prefix << "if " << d.name << " is not None:" << std::endl;
       std::cout << prefix << "  " << d.name << "_tuple = to_matrix_with_info("
           << d.name << ", dtype=np.double, copy=IO.HasParam('copy_all_inputs'))"
           << std::endl;
       std::cout << prefix << "  if len(" << d.name << "_tuple[0].shape"
           << ") < 2:" << std::endl;
       std::cout << prefix << "    " << d.name << "_tuple[0].shape = (" << d.name
           << "_tuple[0].shape[0], 1)" << std::endl;
       std::cout << prefix << "  " << d.name << "_mat = arma_numpy.numpy_to_mat_d("
           << d.name << "_tuple[0], " << d.name << "_tuple[1])" << std::endl;
       std::cout << prefix << "  " << d.name << "_dims = " << d.name
           << "_tuple[2]" << std::endl;
       std::cout << prefix << "  SetParamWithInfo[arma.Mat[double]](<const "
           << "string> '" << d.name << "', dereference(" << d.name << "_mat), "
           << "<const cbool*> " << d.name << "_dims.data)" << std::endl;
       std::cout << prefix << "  IO.SetPassed(<const string> '" << d.name
           << "')" << std::endl;
       std::cout << prefix << "  del " << d.name << "_mat" << std::endl;
     }
     else
     {
       std::cout << prefix << d.name << "_tuple = to_matrix_with_info(" << d.name
           << ", dtype=np.double, copy=IO.HasParam('copy_all_inputs'))"
           << std::endl;
       std::cout << prefix << "if len(" << d.name << "_tuple[0].shape"
           << ") < 2:" << std::endl;
       std::cout << prefix << "  " << d.name << "_tuple[0].shape = (" << d.name
           << "_tuple[0].shape[0], 1)" << std::endl;
       std::cout << prefix << d.name << "_mat = arma_numpy.numpy_to_mat_d("
           << d.name << "_tuple[0], " << d.name << "_tuple[1])" << std::endl;
       std::cout << prefix << d.name << "_dims = " << d.name << "_tuple[2]"
           << std::endl;
       std::cout << prefix << "SetParamWithInfo[arma.Mat[double]](<const "
           << "string> '" << d.name << "', dereference(" << d.name << "_mat), "
           << "<const cbool*> " << d.name << "_dims.data)" << std::endl;
       std::cout << prefix << "IO.SetPassed(<const string> '" << d.name << "')"
           << std::endl;
       std::cout << prefix << "del " << d.name << "_mat" << std::endl;
     }
     std::cout << std::endl;
   }
   
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
