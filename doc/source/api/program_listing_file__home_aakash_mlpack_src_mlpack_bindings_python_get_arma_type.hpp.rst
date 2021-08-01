
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_get_arma_type.hpp:

Program Listing for File get_arma_type.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_get_arma_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/get_arma_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_GET_ARMA_TYPE_HPP
   #define MLPACK_BINDINGS_PYTHON_GET_ARMA_TYPE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   inline std::string GetArmaType()
   {
     if (T::is_col)
       return "col";
     else if (T::is_row)
       return "row";
     else
       return "mat";
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
