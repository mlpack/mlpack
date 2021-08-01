
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_print_input_param.hpp:

Program Listing for File print_input_param.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_print_input_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/print_input_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_JULIA_PRINT_INPUT_PARAM_HPP
   #define MLPACK_BINDINGS_JULIA_PRINT_INPUT_PARAM_HPP
   
   #include "get_julia_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace julia {
   
   template<typename T>
   void PrintInputParam(util::ParamData& d,
                        const void* /* input */,
                        void* /* output */)
   {
     // "type" is a reserved keyword or function.
     const std::string juliaName = (d.name == "type") ? "type_" : d.name;
   
     std::cout << juliaName;
   
     if (!arma::is_arma_type<T>::value)
     {
       std::cout << "::";
       // If it's required, then we need the type.
       if (d.required)
       {
         std::cout << GetJuliaType<typename std::remove_pointer<T>::type>(d);
       }
       else
       {
         std::cout << "Union{"
             << GetJuliaType<typename std::remove_pointer<T>::type>(d)
             << ", Missing} = missing";
       }
     }
     else if (!d.required)
     {
       std::cout << " = missing";
     }
   }
   
   } // namespace julia
   } // namespace bindings
   } // namespace mlpack
   
   #endif
