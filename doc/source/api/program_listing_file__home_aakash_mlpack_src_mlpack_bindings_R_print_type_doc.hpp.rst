
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_type_doc.hpp:

Program Listing for File print_type_doc.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_type_doc.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_type_doc.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_TYPE_DOC_HPP
   #define MLPACK_BINDINGS_R_PRINT_TYPE_DOC_HPP
   
   #include <mlpack/core/util/is_std_vector.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& data,
       const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0);
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& data,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0);
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& data,
       const typename std::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);
   
   template<typename T>
   std::string PrintTypeDoc(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);
   
   template<typename T>
   void PrintTypeDoc(util::ParamData& data,
                     const void* /* input */,
                     void* output)
   {
     *((std::string*) output) =
         PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #include "print_type_doc_impl.hpp"
   
   #endif
