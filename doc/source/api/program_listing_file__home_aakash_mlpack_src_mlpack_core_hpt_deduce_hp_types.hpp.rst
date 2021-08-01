
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_hpt_deduce_hp_types.hpp:

Program Listing for File deduce_hp_types.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_hpt_deduce_hp_types.hpp>` (``/home/aakash/mlpack/src/mlpack/core/hpt/deduce_hp_types.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_HPT_DEDUCE_HP_TYPES_HPP
   #define MLPACK_CORE_HPT_DEDUCE_HP_TYPES_HPP
   
   #include <mlpack/core/hpt/fixed.hpp>
   
   namespace mlpack {
   namespace hpt {
   
   template<typename... Args>
   struct DeduceHyperParameterTypes
   {
     template<typename... HPTypes>
     struct ResultHolder
     {
       using TupleType = std::tuple<HPTypes...>;
     };
   };
   
   template<typename T, typename... Args>
   struct DeduceHyperParameterTypes<T, Args...>
   {
     template<typename ArgumentType,
              bool IsArithmetic = std::is_arithmetic<ArgumentType>::value>
     struct ResultHPType;
   
     template<typename ArithmeticType>
     struct ResultHPType<ArithmeticType, true>
     {
       using Type = ArithmeticType;
     };
   
     template<typename Type>
     struct IsCollectionType
     {
       using Yes = char[1];
       using No = char[2];
   
       template<typename TypeToCheck>
       static Yes& Check(typename TypeToCheck::value_type*);
       template<typename>
       static No& Check(...);
   
       static const bool value  =
         sizeof(decltype(Check<Type>(0))) == sizeof(Yes);
     };
   
     template<typename CollectionType>
     struct ResultHPType<CollectionType, false>
     {
       static_assert(IsCollectionType<CollectionType>::value,
           "One of the passed arguments is neither of an arithmetic type, nor of "
           "a collection type, nor fixed with the Fixed function.");
   
       using Type = typename CollectionType::value_type;
     };
   
     template<typename... HPTypes>
     struct ResultHolder
     {
       using TupleType = typename DeduceHyperParameterTypes<Args...>::template
           ResultHolder<HPTypes..., typename ResultHPType<T>::Type>::TupleType;
     };
   
     using TupleType = typename ResultHolder<>::TupleType;
   };
   
   template<typename T, typename... Args>
   struct DeduceHyperParameterTypes<PreFixedArg<T>, Args...>
   {
     template<typename... HPTypes>
     struct ResultHolder
     {
       using TupleType = typename DeduceHyperParameterTypes<Args...>::template
           ResultHolder<HPTypes...>::TupleType;
     };
   
     using TupleType = typename ResultHolder<>::TupleType;
   };
   
   template<typename... Args>
   using TupleOfHyperParameters =
       typename DeduceHyperParameterTypes<Args...>::TupleType;
   
   } // namespace hpt
   } // namespace mlpack
   
   #endif
