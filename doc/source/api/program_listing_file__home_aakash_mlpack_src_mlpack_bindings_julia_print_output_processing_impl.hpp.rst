
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_print_output_processing_impl.hpp:

Program Listing for File print_output_processing_impl.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_print_output_processing_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/print_output_processing_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP
   #define MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP
   
   #include "print_output_processing.hpp"
   
   #include <mlpack/bindings/util/strip_type.hpp>
   #include "get_julia_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace julia {
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const std::string& /* functionName */,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
       const typename std::enable_if<!data::HasSerialize<T>::value>::type*,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
   {
     std::string type;
     if (std::is_same<T, bool>::value)
       type = "Bool";
     else if (std::is_same<T, int>::value)
       type = "Int";
     else if (std::is_same<T, double>::value)
       type = "Double";
     else if (std::is_same<T, std::string>::value)
       type = "String";
     else if (std::is_same<T, std::vector<std::string>>::value)
       type = "VectorStr";
     else if (std::is_same<T, std::vector<int>>::value)
       type = "VectorInt";
     else
       type = "Unknown";
   
     // Strings need a little special handling.
     if (std::is_same<T, std::string>::value)
       std::cout << "Base.unsafe_string(";
   
     std::cout << "IOGetParam" << type << "(\"" << d.name << "\")";
   
     if (std::is_same<T, std::string>::value)
       std::cout << ")";
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const std::string& /* functionName */,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type*,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
   {
     std::string uChar = (std::is_same<typename T::elem_type, size_t>::value) ?
         "U" : "";
     std::string matTypeSuffix = "";
     std::string extra = "";
     if (T::is_row)
     {
       matTypeSuffix = "Row";
     }
     else if (T::is_col)
     {
       matTypeSuffix = "Col";
     }
     else
     {
       matTypeSuffix = "Mat";
       extra = ", points_are_rows";
     }
   
     std::cout << "IOGetParam" << uChar << matTypeSuffix << "(\"" << d.name
         << "\"" << extra << ")";
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const std::string& functionName,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
       const typename std::enable_if<data::HasSerialize<T>::value>::type*,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
   {
     std::string type = util::StripType(d.cppType);
     std::cout << functionName << "_internal.IOGetParam"
         << type << "(\"" << d.name << "\", modelPtrs)";
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const std::string& /* functionName */,
       const typename std::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
   {
     std::cout << "IOGetParamMatWithInfo(\"" << d.name << "\")";
   }
   
   } // namespace julia
   } // namespace bindings
   } // namespace mlpack
   
   #endif
