
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_test_helper.hpp:

Program Listing for File test_helper.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_test_helper.hpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/test_helper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_TESTS_MAIN_TESTS_TEST_HELPER_HPP
   #define MLPACK_TESTS_MAIN_TESTS_TEST_HELPER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace util {
   
   template<typename T>
   void SetInputParam(const std::string& name, T&& value)
   {
     IO::GetParam<typename std::remove_reference<T>::type>(name) =
       std::forward<T>(value);
     IO::SetPassed(name);
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif
