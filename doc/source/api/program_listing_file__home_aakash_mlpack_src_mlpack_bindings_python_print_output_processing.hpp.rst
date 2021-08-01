
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_print_output_processing.hpp:

Program Listing for File print_output_processing.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_print_output_processing.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/print_output_processing.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_PRINT_OUTPUT_PROCESSING_HPP
   #define MLPACK_BINDINGS_PYTHON_PRINT_OUTPUT_PROCESSING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "get_arma_type.hpp"
   #include "get_numpy_type_char.hpp"
   #include "get_cython_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
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
       std::cout << prefix << "result = " << "IO.GetParam[" << GetCythonType<T>(d)
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
       std::cout << prefix << "result['" << d.name << "'] = IO.GetParam["
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
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const size_t indent,
       const bool onlyOutput,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     if (onlyOutput)
     {
       std::cout << prefix << "result = arma_numpy." << GetArmaType<T>()
           << "_to_numpy_" << GetNumpyTypeChar<T>() << "(IO.GetParam["
           << GetCythonType<T>(d) << "](\"" << d.name << "\"))" << std::endl;
     }
     else
     {
       std::cout << prefix << "result['" << d.name
           << "'] = arma_numpy." << GetArmaType<T>() << "_to_numpy_"
           << GetNumpyTypeChar<T>() << "(IO.GetParam[" << GetCythonType<T>(d)
           << "]('" << d.name << "'))" << std::endl;
     }
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const size_t indent,
       const bool onlyOutput,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     // Print the output with the matrix type.  The dimension information doesn't
     // need to go back.
     if (onlyOutput)
     {
       std::cout << prefix << "result = arma_numpy.mat_to_numpy_"
           << GetNumpyTypeChar<arma::mat>()
           << "(GetParamWithInfo[arma.Mat[double]]('" << d.name << "'))"
           << std::endl;
     }
     else
     {
       std::cout << prefix << "result['" << d.name
           << "'] = arma_numpy.mat_to_numpy_" << GetNumpyTypeChar<arma::mat>()
           << "(GetParamWithInfo[arma.Mat[double]]('" << d.name << "'))"
           << std::endl;
     }
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const size_t indent,
       const bool onlyOutput,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Get the type names we need to use.
     std::string strippedType, printedType, defaultsType;
     StripType(d.cppType, strippedType, printedType, defaultsType);
   
     const std::string prefix(indent, ' ');
   
     if (onlyOutput)
     {
       std::cout << prefix << "result = " << strippedType << "Type()" << std::endl;
       std::cout << prefix << "(<" << strippedType << "Type?> result).modelptr = "
           << "GetParamPtr[" << strippedType << "]('" << d.name << "')"
           << std::endl;
   
       std::map<std::string, util::ParamData>& parameters = IO::Parameters();
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
       std::cout << prefix << "result['" << d.name << "'] = " << strippedType
           << "Type()" << std::endl;
       std::cout << prefix << "(<" << strippedType << "Type?> result['" << d.name
           << "']).modelptr = GetParamPtr[" << strippedType << "]('" << d.name
           << "')" << std::endl;
   
       std::map<std::string, util::ParamData>& parameters = IO::Parameters();
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
   
   template<typename T>
   void PrintOutputProcessing(util::ParamData& d,
                              const void* input,
                              void* /* output */)
   {
     std::tuple<size_t, bool>* tuple = (std::tuple<size_t, bool>*) input;
   
     PrintOutputProcessing<typename std::remove_pointer<T>::type>(d,
         std::get<0>(*tuple), std::get<1>(*tuple));
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
