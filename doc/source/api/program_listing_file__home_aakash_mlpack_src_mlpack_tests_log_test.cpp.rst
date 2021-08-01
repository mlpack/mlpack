
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_log_test.cpp:

Program Listing for File log_test.cpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_log_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/log_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   
   TEST_CASE("LogAssertConditionTest", "[LogTest]")
   {
     // Only do anything for Assert() if in debugging mode.
     #ifdef DEBUG
         // If everything goes well we reach the Catch2 test condition which is
         // always true by simplicity's sake.
         Log::Assert(true, "test");
         REQUIRE(1 == 1);
   
         // The test case should halt the program execution and prints a custom
         // error message. Since the program is halted we should never reach the
         // Catch2 test condition which is always false by simplicity's sake.
         // Log::Assert(false, "test");
         // REQUIRE(1 == 0);
     #endif
   }
