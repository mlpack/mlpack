
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_tests_ignore_check.hpp:

Program Listing for File ignore_check.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_tests_ignore_check.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/tests/ignore_check.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_TEST_IGNORE_CHECK_HPP
   #define MLPACK_BINDINGS_TEST_IGNORE_CHECK_HPP
   
   namespace mlpack {
   namespace bindings {
   namespace tests {
   
   template<typename T>
   inline bool IgnoreCheck(const T& /* t */) { return false; }
   
   } // namespace tests
   } // namespace bindings
   } // namespace mlpack
   
   #endif
