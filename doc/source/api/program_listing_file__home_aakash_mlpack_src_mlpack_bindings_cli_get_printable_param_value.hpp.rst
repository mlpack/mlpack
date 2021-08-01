
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_param_value.hpp:

Program Listing for File get_printable_param_value.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_param_value.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/get_printable_param_value.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_VALUE_HPP
   #define MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_VALUE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/param_data.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& data,
       const std::string& value,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& data,
       const std::string& value,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& data,
       const std::string& value,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& data,
       const std::string& value,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   void GetPrintableParamValue(
       util::ParamData& d,
       const void* input,
       void* output)
   {
     *((std::string*) output) =
         GetPrintableParamValue<typename std::remove_pointer<T>::type>(d,
         *((std::string*) input));
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "get_printable_param_value_impl.hpp"
   
   #endif
