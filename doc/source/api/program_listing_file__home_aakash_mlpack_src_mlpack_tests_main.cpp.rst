
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main.cpp:

Program Listing for File main.cpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <iostream>
   #include <mlpack/core.hpp>
   
   // #define CATCH_CONFIG_MAIN  // catch.hpp will define main()
   #define CATCH_CONFIG_RUNNER  // we will define main()
   #include "catch.hpp"
   
   int main(int argc, char** argv)
   {
     // size_t seed = std::time(NULL);
     // mlpack::math::RandomSeed(seed);
     #ifndef TEST_VERBOSE
       #ifdef DEBUG
       mlpack::Log::Debug.ignoreInput = true;
       #endif
   
       mlpack::Log::Info.ignoreInput = true;
       mlpack::Log::Warn.ignoreInput = true;
     #endif
   
     std::cout << "mlpack version: " << mlpack::util::GetVersion() << std::endl;
     std::cout << "armadillo version: " << arma::arma_version::as_string()
         << std::endl;
   
   
     return Catch::Session().run(argc, argv);
   }
