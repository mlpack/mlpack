
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_print_doc_functions.hpp:

Program Listing for File print_doc_functions.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_print_doc_functions.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/print_doc_functions.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_PRINT_DOC_FUNCTIONS_HPP
   #define MLPACK_BINDINGS_GO_PRINT_DOC_FUNCTIONS_HPP
   
   #include <mlpack/core/util/hyphenate_string.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   inline std::string GetBindingName(const std::string& bindingName);
   
   inline std::string PrintImport();
   
   inline std::string PrintOutputOptionInfo();
   
   inline std::string PrintInputOptionInfo();
   
   template<typename T>
   inline std::string PrintValue(const T& value, bool quotes);
   
   // Special overload for booleans.
   template<>
   inline std::string PrintValue(const bool& value, bool quotes);
   
   inline std::string PrintDefault(const std::string& paramName);
   
   // Base case: no modification needed.
   inline void GetOptions(
       std::vector<std::tuple<std::string, std::string>>& /* results */);
   
   template<typename T, typename... Args>
   void GetOptions(
       std::vector<std::tuple<std::string, std::string>>& results,
       const std::string& paramName,
       const T& value,
       Args... args);
   
   // Recursion base case.
   inline std::string PrintOptionalInputs(/* option */);
   
   // Recursion base case.
   inline std::string PrintInputOptions(/* option */);
   
   template<typename T, typename... Args>
   std::string PrintOptionalInputs(const std::string& paramName,
                                   const T& value,
                                   Args... args);
   
   template<typename T, typename... Args>
   std::string PrintInputOptions(const std::string& paramName,
                                 const T& value,
                                 Args... args);
   
   // Recursion base case.
   inline std::string PrintOutputOptions();
   
   template<typename... Args>
   std::string PrintOutputOptions(Args... args);
   
   template<typename... Args>
   std::string ProgramCall(const std::string& programName, Args... args);
   
   inline std::string PrintModel(const std::string& modelName);
   
   inline std::string PrintDataset(const std::string& datasetName);
   
   inline std::string ParamString(const std::string& paramName);
   
   inline bool IgnoreCheck(const std::string& paramName);
   
   inline bool IgnoreCheck(const std::vector<std::string>& constraints);
   
   inline bool IgnoreCheck(
       const std::vector<std::pair<std::string, bool>>& constraints,
       const std::string& paramName);
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   // Include implementation.
   #include "print_doc_functions_impl.hpp"
   
   #endif
