
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_output_processing.hpp:

Program Listing for File print_output_processing.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_output_processing.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_output_processing.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_OUTPUT_PROCESSING_HPP
   #define MLPACK_BINDINGS_R_PRINT_OUTPUT_PROCESSING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "get_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     MLPACK_COUT_STREAM << "  \"" << d.name << "\" = IO_GetParam" << GetType<T>(d)
               << "(\"" << d.name << "\")";
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
   {
     MLPACK_COUT_STREAM << "  \"" << d.name << "\" = IO_GetParam" << GetType<T>(d)
               << "(\"" << d.name << "\")";
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     MLPACK_COUT_STREAM << "  \"" << d.name << "\" = IO_GetParam" << GetType<T>(d)
               << "(\"" << d.name << "\")";
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     MLPACK_COUT_STREAM << "  \"" << d.name << "\" = " << d.name;
   }
   
   template<typename T>
   void PrintOutputProcessing(util::ParamData& d,
                              const void* /*input*/,
                              void* /* output */)
   {
     PrintOutputProcessing<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
