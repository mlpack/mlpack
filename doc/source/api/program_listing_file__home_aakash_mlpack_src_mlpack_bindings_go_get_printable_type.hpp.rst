
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_get_printable_type.hpp:

Program Listing for File get_printable_type.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_get_printable_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/get_printable_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_GET_PRINTABLE_TYPE_HPP
   #define MLPACK_BINDINGS_GO_GET_PRINTABLE_TYPE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/is_std_vector.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   template<typename T>
   inline std::string GetPrintableType(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<>
   inline std::string GetPrintableType<int>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<int>>::type*,
       const typename boost::disable_if<data::HasSerialize<int>>::type*,
       const typename boost::disable_if<arma::is_arma_type<int>>::type*,
       const typename boost::disable_if<std::is_same<int,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*);
   
   template<>
   inline std::string GetPrintableType<double>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<double>>::type*,
       const typename boost::disable_if<data::HasSerialize<double>>::type*,
       const typename boost::disable_if<arma::is_arma_type<double>>::type*,
       const typename boost::disable_if<std::is_same<double,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*);
   
   template<>
   inline std::string GetPrintableType<std::string>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<std::string>>::type*,
       const typename boost::disable_if<data::HasSerialize<std::string>>::type*,
       const typename boost::disable_if<arma::is_arma_type<std::string>>::type*,
       const typename boost::disable_if<std::is_same<std::string,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*);
   
   template<>
   inline std::string GetPrintableType<bool>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<bool>>::type*,
       const typename boost::disable_if<data::HasSerialize<bool>>::type*,
       const typename boost::disable_if<arma::is_arma_type<bool>>::type*,
       const typename boost::disable_if<std::is_same<bool,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*);
   
   template<typename T>
   inline std::string GetPrintableType(
       util::ParamData& d,
       const typename boost::enable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   inline std::string GetPrintableType(
       util::ParamData& /* d */,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   inline std::string GetPrintableType(
       util::ParamData& /* d */,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   inline std::string GetPrintableType(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   void GetPrintableType(util::ParamData& d,
                         const void* /* input */,
                         void* output)
   {
     *((std::string*) output) =
         GetPrintableType<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #include "get_printable_type_impl.hpp"
   
   #endif
