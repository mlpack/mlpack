
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_printable_param_name_impl.hpp:

Program Listing for File get_printable_param_name_impl.hpp
==========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_printable_param_name_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/get_printable_param_name_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_PARAM_NAME_IMPL_HPP
   #define MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_PARAM_NAME_IMPL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/param_data.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   template<typename T>
   std::string GetPrintableParamName(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::disable_if<data::HasSerialize<T>>::type*,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*)
   {
     return "--" + data.name;
   }
   
   template<typename T>
   std::string GetPrintableParamName(
       util::ParamData& data,
       const typename boost::enable_if<arma::is_arma_type<T>>::type*)
   {
     return "--" + data.name + "_file";
   }
   
   template<typename T>
   std::string GetPrintableParamName(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::enable_if<data::HasSerialize<T>>::type*)
   {
     return "--" + data.name + "_file";
   }
   
   template<typename T>
   std::string GetPrintableParamName(
       util::ParamData& data,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*)
   {
     return "--" + data.name + "_file";
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif
