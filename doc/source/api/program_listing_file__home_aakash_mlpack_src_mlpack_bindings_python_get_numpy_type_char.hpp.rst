
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_get_numpy_type_char.hpp:

Program Listing for File get_numpy_type_char.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_get_numpy_type_char.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/get_numpy_type_char.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_CHAR_HPP
   #define MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_CHAR_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   inline std::string GetNumpyTypeChar()
   {
     return "?";
   }
   
   // size_t = s.
   template<>
   inline std::string GetNumpyTypeChar<arma::Mat<size_t>>()
   {
     return "s";
   }
   
   template<>
   inline std::string GetNumpyTypeChar<arma::Col<size_t>>()
   {
     return "s";
   }
   
   template<>
   inline std::string GetNumpyTypeChar<arma::Row<size_t>>()
   {
     return "s";
   }
   
   // double = d.
   template<>
   inline std::string GetNumpyTypeChar<arma::mat>()
   {
     return "d";
   }
   
   template<>
   inline std::string GetNumpyTypeChar<arma::vec>()
   {
     return "d";
   }
   
   template<>
   inline std::string GetNumpyTypeChar<arma::rowvec>()
   {
     return "d";
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
