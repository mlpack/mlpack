
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_string_type_param_impl.hpp:

Program Listing for File string_type_param_impl.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_string_type_param_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/string_type_param_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_STRING_TYPE_PARAM_IMPL_HPP
   #define MLPACK_BINDINGS_CLI_STRING_TYPE_PARAM_IMPL_HPP
   
   #include "string_type_param.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   std::string StringTypeParamImpl(
       const typename boost::disable_if<util::IsStdVector<T>>::type* /* junk */,
       const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */)
   {
     // Don't know what type this is.
     return "unknown";
   }
   
   template<typename T>
   std::string StringTypeParamImpl(
       const typename boost::enable_if<util::IsStdVector<T>>::type* /* junk */)
   {
     return "vector";
   }
   
   template<typename T>
   std::string StringTypeParamImpl(
       const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
   {
     return "string";
   }
   
   template<>
   inline void StringTypeParam<int>(util::ParamData& /* data */,
                                    const void* /* input */,
                                    void* output)
   {
     std::string* outstr = (std::string*) output;
     *outstr = "int";
   }
   
   template<>
   inline void StringTypeParam<bool>(util::ParamData& /* data */,
                                     const void* /* input */,
                                     void* output)
   {
     std::string* outstr = (std::string*) output;
     *outstr = "bool";
   }
   
   template<>
   inline void StringTypeParam<std::string>(util::ParamData& /* data */,
                                            const void* /* input */,
                                            void* output)
   {
     std::string* outstr = (std::string*) output;
     *outstr = "string";
   }
   
   template<>
   inline void StringTypeParam<double>(util::ParamData& /* data */,
                                       const void* /* input */,
                                       void* output)
   {
     std::string* outstr = (std::string*) output;
     *outstr = "double";
   }
   
   template<>
   inline void StringTypeParam<std::tuple<mlpack::data::DatasetInfo, arma::mat>>(
       util::ParamData& /* data */,
       const void* /* input */,
       void* output)
   {
     std::string* outstr = (std::string*) output;
     *outstr = "string";
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
