
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_doc_functions.hpp:

Program Listing for File print_doc_functions.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_doc_functions.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_doc_functions.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_DOC_FUNCTIONS_HPP
   #define MLPACK_BINDINGS_R_PRINT_DOC_FUNCTIONS_HPP
   
   #include <mlpack/core/util/hyphenate_string.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   inline std::string GetBindingName(const std::string& bindingName);
   
   inline std::string PrintImport();
   
   inline std::string PrintInputOptionInfo();
   
   inline std::string PrintOutputOptionInfo();
   
   template<typename T>
   inline std::string PrintValue(const T& value, bool quotes);
   
   template<>
   inline std::string PrintValue(const bool& value, bool quotes);
   
   inline std::string PrintDefault(const std::string& paramName);
   
   inline std::string PrintInputOptions();
   
   template<typename T, typename... Args>
   std::string PrintInputOptions(const std::string& paramName,
                                 const T& value,
                                 Args... args);
   
   inline std::string PrintOutputOptions(const bool /* markdown */);
   
   template<typename T, typename... Args>
   std::string PrintOutputOptions(const bool markdown,
                                  const std::string& paramName,
                                  const T& value,
                                  Args... args);
   
   template<typename... Args>
   std::string ProgramCall(const bool markdown,
                           const std::string& programName,
                           Args... args);
   
   inline std::string ProgramCall(const std::string& programName);
   
   inline std::string PrintModel(const std::string& modelName);
   
   inline std::string PrintDataset(const std::string& datasetName);
   
   inline std::string ParamString(const std::string& paramName);
   
   inline bool IgnoreCheck(const std::string& paramName);
   
   inline bool IgnoreCheck(const std::vector<std::string>& constraints);
   
   inline bool IgnoreCheck(
       const std::vector<std::pair<std::string, bool>>& constraints,
       const std::string& paramName);
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "print_doc_functions_impl.hpp"
   
   #endif
