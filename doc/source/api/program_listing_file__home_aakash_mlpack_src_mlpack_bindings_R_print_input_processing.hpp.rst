
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_input_processing.hpp:

Program Listing for File print_input_processing.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_input_processing.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_input_processing.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_INPUT_PROCESSING_IMPL_HPP
   #define MLPACK_BINDINGS_R_PRINT_INPUT_PROCESSING_IMPL_HPP
   
   #include <mlpack/bindings/util/strip_type.hpp>
   #include "get_type.hpp"
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     if (!d.required)
     {
       MLPACK_COUT_STREAM << "  if (!identical(" << d.name;
       if (d.cppType == "bool")
       {
         MLPACK_COUT_STREAM << ", FALSE)) {" << std::endl;
       }
       else
       {
         MLPACK_COUT_STREAM << ", NA)) {" << std::endl;
       }
       MLPACK_COUT_STREAM << "    IO_SetParam" << GetType<T>(d) << "(\""
           << d.name << "\", " << d.name << ")" << std::endl;
       MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
     }
     else
     {
       MLPACK_COUT_STREAM << "  IO_SetParam" << GetType<T>(d) << "(\""
                 << d.name << "\", " << d.name << ")" << std::endl;
     }
     MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
   }
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     if (!d.required)
     {
       MLPACK_COUT_STREAM << "  if (!identical(" << d.name << ", NA)) {"
           << std::endl;
       MLPACK_COUT_STREAM << "    IO_SetParam" << GetType<T>(d) << "(\""
           << d.name << "\", to_matrix(" << d.name << "))" << std::endl;
       MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
     }
     else
     {
       MLPACK_COUT_STREAM << "  IO_SetParam" << GetType<T>(d) << "(\""
           << d.name << "\", to_matrix(" << d.name << "))" << std::endl;
     }
     MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
   }
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     if (!d.required)
     {
       MLPACK_COUT_STREAM << "  if (!identical(" << d.name << ", NA)) {"
           << std::endl;
       MLPACK_COUT_STREAM << "    " << d.name << " <- to_matrix_with_info("
           << d.name << ")" << std::endl;
       MLPACK_COUT_STREAM << "    IO_SetParam" << GetType<T>(d) << "(\""
           << d.name << "\", " << d.name << "$info, " << d.name
           << "$data)" << std::endl;
       MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
     }
     else
     {
       MLPACK_COUT_STREAM << "  " << d.name << " <- to_matrix_with_info("
           << d.name << ")" << std::endl;
       MLPACK_COUT_STREAM << "  IO_SetParam" << GetType<T>(d) << "(\""
           << d.name << "\", " << d.name << "$info, " << d.name
           << "$data)" << std::endl;
     }
     MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
   }
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     if (!d.required)
     {
       MLPACK_COUT_STREAM << "  if (!identical(" << d.name << ", NA)) {"
           << std::endl;
       MLPACK_COUT_STREAM << "    IO_SetParam" << util::StripType(d.cppType)
           << "Ptr(\"" << d.name << "\", " << d.name << ")" << std::endl;
       MLPACK_COUT_STREAM << "  }" << std::endl; // Closing brace.
     }
     else
     {
       MLPACK_COUT_STREAM << "  IO_SetParam" << util::StripType(d.cppType)
           << "Ptr(\"" << d.name << "\", " << d.name << ")" << std::endl;
     }
     MLPACK_COUT_STREAM << std::endl; // Extra line is to clear up the code a bit.
   }
   
   template<typename T>
   void PrintInputProcessing(util::ParamData& d,
                             const void* /* input */,
                             void* /* output */)
   {
     PrintInputProcessing<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
