
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_R.hpp:

Program Listing for File print_R.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_R.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_R.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_R_HPP
   #define MLPACK_BINDINGS_R_PRINT_R_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   void PrintR(const util::BindingDetails& doc,
               const std::string& functionName);
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
