
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_print_go.hpp:

Program Listing for File print_go.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_print_go.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/print_go.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_PRINT_GO_HPP
   #define MLPACK_BINDINGS_GO_PRINT_GO_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   void PrintGo(const util::BindingDetails& doc,
                const std::string& functionName);
   
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif
