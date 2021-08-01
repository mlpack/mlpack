
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_output_param.hpp:

Program Listing for File output_param.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_output_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/output_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_OUTPUT_PARAM_HPP
   #define MLPACK_BINDINGS_CLI_OUTPUT_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/param_data.hpp>
   #include <mlpack/core/util/is_std_vector.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   void OutputParamImpl(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   void OutputParamImpl(
       util::ParamData& data,
       const typename boost::enable_if<util::IsStdVector<T>>::type* = 0);
   
   template<typename T>
   void OutputParamImpl(
       util::ParamData& data,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);
   
   template<typename T>
   void OutputParamImpl(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);
   
   template<typename T>
   void OutputParamImpl(
       util::ParamData& data,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);
   
   template<typename T>
   void OutputParam(util::ParamData& data,
                    const void* /* input */,
                    void* /* output */)
   {
     OutputParamImpl<typename std::remove_pointer<T>::type>(data);
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "output_param_impl.hpp"
   
   #endif
