
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_print_jl.hpp:

Program Listing for File print_jl.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_print_jl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/print_jl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_JULIA_PRINT_JL_HPP
   #define MLPACK_BINDINGS_JULIA_PRINT_JL_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace julia {
   
   void PrintJL(const util::BindingDetails& doc,
                const std::string& functionName,
                const std::string& mlpackJuliaLibSuffix);
   
   } // namespace julia
   } // namespace bindings
   } // namespace mlpack
   
   #endif
