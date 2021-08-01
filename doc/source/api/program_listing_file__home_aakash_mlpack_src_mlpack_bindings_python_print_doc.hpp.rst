
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_print_doc.hpp:

Program Listing for File print_doc.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_print_doc.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/print_doc.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_PRINT_DOC_HPP
   #define MLPACK_BINDINGS_PYTHON_PRINT_DOC_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/hyphenate_string.hpp>
   #include "get_printable_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   void PrintDoc(util::ParamData& d,
                 const void* input,
                 void* /* output */)
   {
     const size_t indent = *((size_t*) input);
     std::ostringstream oss;
     oss << " - ";
     if (d.name == "lambda") // Don't print Python keywords.
       oss << d.name << "_ (";
     else
       oss << d.name << " (";
     oss << GetPrintableType<typename std::remove_pointer<T>::type>(d) << "): "
         << d.desc;
   
     // Print a default, if possible.
     if (!d.required)
     {
       // Call the correct overload to get the default value directly.
       if (d.cppType == "std::string" || d.cppType == "double" ||
           d.cppType == "int" || d.cppType == "std::vector<int>" ||
           d.cppType == "std::vector<std::string>" ||
           d.cppType == "std::vector<double>")
       {
         std::string defaultValue = DefaultParamImpl<T>(d);
         oss << "  Default value " << defaultValue << ".";
       }
     }
   
     std::cout << util::HyphenateString(oss.str(), indent + 4);
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
