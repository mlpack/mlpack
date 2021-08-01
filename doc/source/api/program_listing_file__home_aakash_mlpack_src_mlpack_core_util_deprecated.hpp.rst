
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_deprecated.hpp:

Program Listing for File deprecated.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_deprecated.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/deprecated.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_DEPRECATED_HPP
   #define MLPACK_CORE_UTIL_DEPRECATED_HPP
   
   #ifdef __GNUG__
   #define mlpack_deprecated __attribute__((deprecated))
   #elif defined(_MSC_VER)
   #define mlpack_deprecated __declspec(deprecated)
   #else
   #pragma message("WARNING: You need to implement mlpack_deprecated for this "
       "compiler")
   #define mlpack_deprecated
   #endif
   
   #endif
