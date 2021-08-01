
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_hpt_fixed.hpp:

Program Listing for File fixed.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_hpt_fixed.hpp>` (``/home/aakash/mlpack/src/mlpack/core/hpt/fixed.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_HPT_FIXED_HPP
   #define MLPACK_CORE_HPT_FIXED_HPP
   
   #include <type_traits>
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace hpt {
   
   template<typename>
   struct PreFixedArg;
   
   template<typename T>
   PreFixedArg<T> Fixed(T&& value)
   {
     return PreFixedArg<T>{std::forward<T>(value)};
   }
   
   template<typename T, size_t I>
   struct FixedArg
   {
     static const size_t index = I;
   
     const T& value;
   };
   
   template<typename T>
   struct PreFixedArg
   {
     using Type = T;
   
     const T value;
   };
   
   template<typename T>
   struct PreFixedArg<T&>
   {
     using Type = T;
   
     const T& value;
   };
   
   template<typename T>
   class IsPreFixedArg
   {
     template<typename>
     struct Implementation : std::false_type {};
   
     template<typename Type>
     struct Implementation<PreFixedArg<Type>> : std::true_type {};
   
    public:
     static const bool value = Implementation<typename std::decay<T>::type>::value;
   };
   
   } // namespace hpt
   } // namespace mlpack
   
   #endif
