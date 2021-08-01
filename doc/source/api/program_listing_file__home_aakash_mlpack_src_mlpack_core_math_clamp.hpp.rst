
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_clamp.hpp:

Program Listing for File clamp.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_clamp.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/clamp.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_CLAMP_HPP
   #define MLPACK_CORE_MATH_CLAMP_HPP
   
   #include <stdlib.h>
   #include <math.h>
   #include <float.h>
   
   namespace mlpack {
   namespace math  {
   
   inline double ClampNonNegative(const double d)
   {
     return (d + fabs(d)) / 2;
   }
   
   inline double ClampNonPositive(const double d)
   {
     return (d - fabs(d)) / 2;
   }
   
   inline double ClampRange(double value,
                            const double rangeMin,
                            const double rangeMax)
   {
     value -= rangeMax;
     value = ClampNonPositive(value) + rangeMax;
     value -= rangeMin;
     value = ClampNonNegative(value) + rangeMin;
     return value;
   }
   
   } // namespace math
   } // namespace mlpack
   
   #endif // MLPACK_CORE_MATH_CLAMP_HPP
