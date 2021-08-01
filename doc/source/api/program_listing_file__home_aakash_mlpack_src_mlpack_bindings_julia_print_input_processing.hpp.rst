
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_print_input_processing.hpp:

Program Listing for File print_input_processing.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_print_input_processing.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/print_input_processing.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_JULIA_PRINT_INPUT_PROCESSING_HPP
   #define MLPACK_BINDINGS_JULIA_PRINT_INPUT_PROCESSING_HPP
   
   namespace mlpack {
   namespace bindings {
   namespace julia {
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const std::string& functionName,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const std::string& functionName,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const std::string& functionName,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);
   
   template<typename T>
   void PrintInputProcessing(
       util::ParamData& d,
       const std::string& functionName,
       const typename std::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);
   
   template<typename T>
   void PrintInputProcessing(util::ParamData& d,
                             const void* input,
                             void* /* output */)
   {
     // Call out to the right overload.
     PrintInputProcessing<typename std::remove_pointer<T>::type>(d,
         *((std::string*) input));
   }
   
   } // namespace julia
   } // namespace bindings
   } // namespace mlpack
   
   #include "print_input_processing_impl.hpp"
   
   #endif
