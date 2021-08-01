
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_input_param.hpp:

Program Listing for File print_input_param.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_input_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_input_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_INPUT_PARAM_HPP
   #define MLPACK_BINDINGS_R_PRINT_INPUT_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   void PrintInputParam(util::ParamData& d,
                        const void* /* input */,
                        void* /* output */)
   {
     MLPACK_COUT_STREAM << d.name;
     if (std::is_same<T, bool>::value)
       MLPACK_COUT_STREAM << "=FALSE";
     else if (!d.required)
       MLPACK_COUT_STREAM << "=NA";
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
