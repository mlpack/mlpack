
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_print_type_doc.hpp:

Program Listing for File print_type_doc.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_print_type_doc.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/print_type_doc.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_TYPE_DOC_HPP
   #define MLPACK_BINDINGS_MARKDOWN_PRINT_TYPE_DOC_HPP
   
   #include "binding_info.hpp"
   
   #include <mlpack/bindings/cli/print_type_doc.hpp>
   #include <mlpack/bindings/python/print_type_doc.hpp>
   #include <mlpack/bindings/julia/print_type_doc.hpp>
   #include <mlpack/bindings/go/print_type_doc.hpp>
   #include <mlpack/bindings/R/print_type_doc.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   template<typename T>
   std::string PrintTypeDoc(util::ParamData& data)
   {
     if (BindingInfo::Language() == "cli")
     {
       return cli::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "python")
     {
       return python::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "julia")
     {
       return julia::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "go")
     {
       return go::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
     }
     else if (BindingInfo::Language() == "r")
     {
       return r::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
     }
     else
     {
       throw std::invalid_argument("PrintTypeDoc(): unknown "
           "BindingInfo::Language() " + BindingInfo::Language() + "!");
     }
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif
