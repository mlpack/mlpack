
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_default_param.hpp:

Program Listing for File default_param.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_default_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/default_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_DEFAULT_PARAM_HPP
   #define MLPACK_BINDINGS_MARKDOWN_DEFAULT_PARAM_HPP
   
   #include "binding_info.hpp"
   
   #include <mlpack/bindings/cli/default_param.hpp>
   #include <mlpack/bindings/python/default_param.hpp>
   #include <mlpack/bindings/julia/default_param.hpp>
   #include <mlpack/bindings/go/default_param.hpp>
   #include <mlpack/bindings/R/default_param.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   template<typename T>
   void DefaultParam(util::ParamData& data,
                     const void* /* input */,
                     void* output)
   {
     if (BindingInfo::Language() == "cli")
     {
       *((std::string*) output) =
           cli::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "python")
     {
       *((std::string*) output) =
           python::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "julia")
     {
       *((std::string*) output) =
           julia::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "go")
     {
       *((std::string*) output) =
           go::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "r")
     {
       *((std::string*) output) =
           r::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
     }
     else
     {
       throw std::invalid_argument("DefaultParam(): unknown "
           "BindingInfo::Language() " + BindingInfo::Language() + "!");
     }
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif
