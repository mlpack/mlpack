
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_print_doc_functions.hpp:

Program Listing for File print_doc_functions.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_print_doc_functions.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/print_doc_functions.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_DOC_FUNCTIONS_HPP
   #define MLPACK_BINDINGS_MARKDOWN_PRINT_DOC_FUNCTIONS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   inline std::string GetBindingName(const std::string& bindingName);
   
   inline std::string PrintLanguage(const std::string& language);
   
   inline std::string PrintImport(const std::string& bindingName);
   
   inline std::string PrintInputOptionInfo(const std::string& language);
   
   inline std::string PrintOutputOptionInfo(const std::string& language);
   
   inline std::string PrintTypeDocs();
   
   template<typename T>
   inline std::string PrintValue(const T& value, bool quotes);
   
   inline std::string PrintDefault(const std::string& paramName);
   
   inline std::string PrintDataset(const std::string& dataset);
   
   inline std::string PrintModel(const std::string& model);
   
   template<typename... Args>
   std::string ProgramCall(const std::string& programName, Args... args);
   
   inline std::string ProgramCall(const std::string& programName);
   
   inline std::string ParamString(const std::string& paramName);
   
   inline std::string ParamType(util::ParamData& d);
   
   template<typename T>
   inline bool IgnoreCheck(const T& t);
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "print_doc_functions_impl.hpp"
   
   #endif
