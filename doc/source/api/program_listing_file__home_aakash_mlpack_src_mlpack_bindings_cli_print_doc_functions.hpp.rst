
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_print_doc_functions.hpp:

Program Listing for File print_doc_functions.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_print_doc_functions.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/print_doc_functions.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_HPP
   #define MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_HPP
   
   #include <mlpack/core/util/hyphenate_string.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   inline std::string GetBindingName(const std::string& bindingName);
   
   inline std::string PrintImport(const std::string& bindingName);
   
   inline std::string PrintInputOptionInfo();
   
   inline std::string PrintOutputOptionInfo();
   
   inline std::string PrintTypeDocs();
   
   template<typename T>
   inline std::string PrintValue(const T& value, bool quotes);
   
   inline std::string PrintDefault(const std::string& paramName);
   
   inline std::string PrintDataset(const std::string& dataset);
   
   inline std::string PrintModel(const std::string& model);
   
   inline std::string PrintType(util::ParamData& param);
   
   inline std::string ProcessOptions();
   
   template<typename T, typename... Args>
   std::string ProcessOptions(const std::string& paramName,
                              const T& value,
                              Args... args);
   
   template<typename... Args>
   std::string ProgramCall(const std::string& programName, Args... args);
   
   inline std::string ProgramCall(const std::string& programName);
   
   inline std::string ParamString(const std::string& paramName);
   
   template<typename T>
   inline bool IgnoreCheck(const T& /* t */) { return false; }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "print_doc_functions_impl.hpp"
   
   #endif
