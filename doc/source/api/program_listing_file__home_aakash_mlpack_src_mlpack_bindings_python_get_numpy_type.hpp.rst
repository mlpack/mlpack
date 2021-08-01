
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_get_numpy_type.hpp:

Program Listing for File get_numpy_type.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_get_numpy_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/get_numpy_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_HPP
   #define MLPACK_BINDINGS_PYTHON_GET_NUMPY_TYPE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   inline std::string GetNumpyType()
   {
     return "unknown"; // Not sure...
   }
   
   template<>
   inline std::string GetNumpyType<double>()
   {
     return "np.double";
   }
   
   template<>
   inline std::string GetNumpyType<size_t>()
   {
     return "np.intp";
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
