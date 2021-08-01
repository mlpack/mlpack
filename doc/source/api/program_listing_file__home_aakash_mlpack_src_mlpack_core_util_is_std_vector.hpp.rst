
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_is_std_vector.hpp:

Program Listing for File is_std_vector.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_is_std_vector.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/is_std_vector.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_IS_STD_VECTOR_HPP
   #define MLPACK_CORE_UTIL_IS_STD_VECTOR_HPP
   
   #include <vector>
   
   namespace mlpack {
   namespace util {
   
   template<typename T>
   struct IsStdVector { const static bool value = false; };
   
   template<typename T, typename A>
   struct IsStdVector<std::vector<T, A>> { const static bool value = true; };
   
   } // namespace util
   } // namespace mlpack
   
   #endif
