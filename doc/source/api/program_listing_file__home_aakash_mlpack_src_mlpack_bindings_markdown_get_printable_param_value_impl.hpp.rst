
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_printable_param_value_impl.hpp:

Program Listing for File get_printable_param_value_impl.hpp
===========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_printable_param_value_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/get_printable_param_value_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_PARAM_VALUE_IMPL_HPP
   #define MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_PARAM_VALUE_IMPL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/param_data.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& /* data */,
       const std::string& input,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::disable_if<data::HasSerialize<T>>::type*,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*)
   {
     return input;
   }
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& /* data */,
       const std::string& input,
       const typename boost::enable_if<arma::is_arma_type<T>>::type*)
   {
     return input + ".csv";
   }
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& /* data */,
       const std::string& input,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::enable_if<data::HasSerialize<T>>::type*)
   {
     return input + ".bin";
   }
   
   template<typename T>
   std::string GetPrintableParamValue(
       util::ParamData& /* data */,
       const std::string& input,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*)
   {
     return input + ".arff";
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif
