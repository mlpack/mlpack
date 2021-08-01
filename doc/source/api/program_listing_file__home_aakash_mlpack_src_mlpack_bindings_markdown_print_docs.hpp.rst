
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_print_docs.hpp:

Program Listing for File print_docs.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_print_docs.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/print_docs.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_DOCS_HPP
   #define MLPACK_BINDINGS_MARKDOWN_PRINT_DOCS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   void PrintHeaders(const std::string& bindingName,
                     const std::vector<std::string>& languages);
   
   void PrintDocs(const std::string& bindingName,
                  const std::vector<std::string>& languages);
   
   #endif
