
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_print_doc.hpp:

Program Listing for File print_doc.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_print_doc.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/print_doc.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_PRINT_DOC_HPP
   #define MLPACK_BINDINGS_GO_PRINT_DOC_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/hyphenate_string.hpp>
   #include "get_go_type.hpp"
   #include <mlpack/bindings/util/camel_case.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   template<typename T>
   void PrintDoc(util::ParamData& d,
                 const void* input,
                 void* isLower)
   {
     const size_t indent = *((size_t*) input);
     bool Lower = *((bool*) isLower);
     std::ostringstream oss;
     oss << " - ";
     oss << util::CamelCase(d.name, Lower) << " (";
     oss << GetGoType<typename std::remove_pointer<T>::type>(d) << "): "
         << d.desc;
   
     // Print a default, if possible.
     if (!d.required)
     {
       if (d.cppType == "std::string")
       {
         oss << "  Default value '" << boost::any_cast<std::string>(d.value)
             << "'.";
       }
       else if (d.cppType == "double")
       {
         oss << "  Default value " << boost::any_cast<double>(d.value) << ".";
       }
       else if (d.cppType == "int")
       {
         oss << "  Default value " << boost::any_cast<int>(d.value) << ".";
       }
     }
   
     std::cout << util::HyphenateString(oss.str(), indent + 4);
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif
