
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_print_model_type_import.hpp:

Program Listing for File print_model_type_import.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_print_model_type_import.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/print_model_type_import.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_JULIA_PRINT_MODEL_TYPE_IMPORT_HPP
   #define MLPACK_BINDINGS_JULIA_PRINT_MODEL_TYPE_IMPORT_HPP
   
   #include <mlpack/bindings/util/strip_type.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace julia {
   
   template<typename T>
   void PrintModelTypeImport(
       util::ParamData& /* d */,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
   {
     // Do nothing.
   }
   
   template<typename T>
   void PrintModelTypeImport(
       util::ParamData& /* d */,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
   {
     // Do nothing.
   }
   
   template<typename T>
   void PrintModelTypeImport(
       util::ParamData& d,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
   {
     // We need to print, e.g.,
     // import ..<type>
   
     // First, print the struct definition.
     std::cout << "import .." << util::StripType(d.cppType) << std::endl;
   }
   
   template<typename T>
   void PrintModelTypeImport(util::ParamData& d,
                             const void* /* input */,
                             void* /* output */)
   {
     PrintModelTypeImport<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace julia
   } // namespace bindings
   } // namespace mlpack
   
   #endif
