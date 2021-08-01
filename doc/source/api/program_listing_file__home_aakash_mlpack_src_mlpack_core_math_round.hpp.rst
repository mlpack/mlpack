
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_round.hpp:

Program Listing for File round.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_round.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/round.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_ROUND_HPP
   #define MLPACK_CORE_MATH_ROUND_HPP
   
   // _MSC_VER should only be defined for Visual Studio, which doesn't implement
   // C99.
   #ifdef _MSC_VER
   
   // This function ends up going into the global namespace, so it can be used in
   // place of C99's round().
   
   inline double round(double a)
   {
     return floor(a + 0.5);
   }
   
   #endif
   
   #endif
