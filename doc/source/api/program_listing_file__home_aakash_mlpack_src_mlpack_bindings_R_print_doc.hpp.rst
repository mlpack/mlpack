
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_doc.hpp:

Program Listing for File print_doc.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_doc.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_doc.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_DOC_HPP
   #define MLPACK_BINDINGS_R_PRINT_DOC_HPP
   
   #include "get_r_type.hpp"
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/hyphenate_string.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   void PrintDoc(util::ParamData& d,
                 const void* /* input */,
                 void* output)
   {
     bool out = *((bool*) output);
     std::ostringstream oss;
     if (out)
       oss << "#' \\item{" << d.name << "}{";
     else
       oss << "#' @param " << d.name << " ";
     oss << d.desc.substr(0, d.desc.size() - 1);
     // Print a default, if possible.
     if (!d.required)
     {
       if (d.cppType == "std::string" ||
           d.cppType == "double" ||
           d.cppType == "int" ||
           d.cppType == "bool")
       {
         oss << ".  Default value \"";
         if (d.cppType == "std::string")
         {
           oss << boost::any_cast<std::string>(d.value);
         }
         else if (d.cppType == "double")
         {
           oss << boost::any_cast<double>(d.value);
         }
         else if (d.cppType == "int")
         {
           oss << boost::any_cast<int>(d.value);
         }
         else if (d.cppType == "bool")
         {
           oss << (boost::any_cast<bool>(d.value) ? "TRUE" : "FALSE");
         }
         oss << "\"";
       }
     }
   
     oss << " (" << GetRType<typename std::remove_pointer<T>::type>(d) << ").";
   
     if (out)
       oss << "}";
   
     MLPACK_COUT_STREAM << util::HyphenateString(oss.str(), "#'   ");
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
