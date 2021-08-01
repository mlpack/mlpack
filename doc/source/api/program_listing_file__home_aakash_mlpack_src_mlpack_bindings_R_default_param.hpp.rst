
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_default_param.hpp:

Program Listing for File default_param.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_default_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/default_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_DEFAULT_PARAM_HPP
   #define MLPACK_BINDINGS_R_DEFAULT_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/param_data.hpp>
   #include <mlpack/core/util/is_std_vector.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T, std::string>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::enable_if<util::IsStdVector<T>>::type* = 0);
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::enable_if<std::is_same<T, std::string>>::type* = 0);
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::enable_if_c<
           arma::is_arma_type<T>::value ||
           std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                      arma::mat>>::value>::type* /* junk */ = 0);
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);
   
   template<typename T>
   void DefaultParam(util::ParamData& data,
                     const void* /* input */,
                     void* output)
   {
     std::string* outstr = (std::string*) output;
     *outstr = DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "default_param_impl.hpp"
   
   #endif
