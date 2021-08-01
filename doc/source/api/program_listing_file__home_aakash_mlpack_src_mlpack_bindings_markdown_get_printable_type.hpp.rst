
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_printable_type.hpp:

Program Listing for File get_printable_type.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_printable_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/get_printable_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_TYPE_HPP
   #define MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_TYPE_HPP
   
   #include "binding_info.hpp"
   
   #include <mlpack/bindings/cli/get_printable_type.hpp>
   #include <mlpack/bindings/python/get_printable_type.hpp>
   #include <mlpack/bindings/julia/get_printable_type.hpp>
   #include <mlpack/bindings/go/get_printable_type.hpp>
   #include <mlpack/bindings/R/get_printable_type.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   template<typename T>
   void GetPrintableType(util::ParamData& data,
                         const void* /* input */,
                         void* output)
   {
     if (BindingInfo::Language() == "cli")
     {
       *((std::string*) output) =
           cli::GetPrintableType<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "python")
     {
       *((std::string*) output) =
           python::GetPrintableType<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "julia")
     {
       *((std::string*) output) =
           julia::GetPrintableType<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "go")
     {
       *((std::string*) output) =
           go::GetPrintableType<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "r")
     {
       *((std::string*) output) =
           r::GetPrintableType<typename std::remove_pointer<T>::type>(data);
     }
     else
     {
       throw std::invalid_argument("GetPrintableType(): unknown "
           "BindingInfo::Language(): " + BindingInfo::Language() + "!");
     }
   }
   
   template<typename T>
   std::string GetPrintableType(util::ParamData& data)
   {
     std::string output;
     GetPrintableType<T>(data, (void*) NULL, (void*) &output);
     return output;
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif
