
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_type.hpp:

Program Listing for File get_printable_type.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/get_printable_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_HPP
   #define MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_HPP
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0);
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0);
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename std::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);
   
   template<typename T>
   void GetPrintableType(util::ParamData& data,
                          const void* /* input */,
                          void* output)
   {
     *((std::string*) output) =
         GetPrintableType<typename std::remove_pointer<T>::type>(data);
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #include "get_printable_type_impl.hpp"
   
   #endif
