
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_string_type_param.hpp:

Program Listing for File string_type_param.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_string_type_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/string_type_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_STRING_TYPE_PARAM_HPP
   #define MLPACK_BINDINGS_CLI_STRING_TYPE_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/param_data.hpp>
   #include <mlpack/core/util/is_std_vector.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   std::string StringTypeParamImpl(
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0);
   
   template<typename T>
   std::string StringTypeParamImpl(
       const typename boost::enable_if<util::IsStdVector<T>>::type* = 0);
   
   template<typename T>
   std::string StringTypeParamImpl(
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);
   
   template<typename T>
   void StringTypeParam(util::ParamData& /* data */,
                        const void* /* input */,
                        void* output)
   {
     std::string* outstr = (std::string*) output;
     *outstr = StringTypeParamImpl<T>();
   }
   
   template<>
   inline void StringTypeParam<int>(util::ParamData& /* data */,
                                    const void* /* input */,
                                    void* output);
   
   template<>
   inline void StringTypeParam<bool>(util::ParamData& /* data */,
                                     const void* /* input */,
                                     void* output);
   
   template<>
   inline void StringTypeParam<std::string>(util::ParamData& /* data */,
                                            const void* /* input */,
                                            void* output);
   
   template<>
   inline void StringTypeParam<double>(util::ParamData& /* data */,
                                       const void* /* input */,
                                       void* output);
   
   template<>
   inline void StringTypeParam<std::tuple<mlpack::data::DatasetInfo, arma::mat>>(
       util::ParamData& /* data */,
       const void* /* input */,
       void* output);
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "string_type_param_impl.hpp"
   
   #endif
