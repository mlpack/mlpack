
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_vector_variant_wrapper.hpp:

Program Listing for File pointer_vector_variant_wrapper.hpp
===========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_vector_variant_wrapper.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cereal/pointer_vector_variant_wrapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CEREAL_POINTER_VECTOR_VARIANT_WRAPPER_HPP
   #define MLPACK_CORE_CEREAL_POINTER_VECTOR_VARIANT_WRAPPER_HPP
   
   #include "pointer_wrapper.hpp"
   #include "pointer_variant_wrapper.hpp"
   #include "pointer_vector_wrapper.hpp"
   
   namespace cereal {
   
   // Forward declaration
   template<typename... VariantTypes>
   class PointerVectorVariantWrapper;
   
   template<typename... VariantTypes>
   inline PointerVectorVariantWrapper<VariantTypes...>
   make_vector_pointer_variant(std::vector<boost::variant<VariantTypes...>>& t)
   {
     return PointerVectorVariantWrapper<VariantTypes...>(t);
   }
   
   template<typename... VariantTypes>
   class PointerVectorVariantWrapper
   {
    public:
     PointerVectorVariantWrapper(
         std::vector<boost::variant<VariantTypes...>>& vecPointerVar)
         : vectorPointerVariant(vecPointerVar)
     {}
   
     template<class Archive>
     void save(Archive& ar) const
     {
       size_t vecSize = vectorPointerVariant.size();
       ar(CEREAL_NVP(vecSize));
       for (size_t i = 0; i < vectorPointerVariant.size(); ++i)
       {
         ar(CEREAL_VARIANT_POINTER(vectorPointerVariant.at(i)));
       }
     }
   
     template<class Archive>
     void load(Archive& ar)
     {
       size_t vecSize = 0;
       ar(CEREAL_NVP(vecSize));
       vectorPointerVariant.resize(vecSize);
       for (size_t i = 0; i < vectorPointerVariant.size(); ++i)
       {
         ar(CEREAL_VARIANT_POINTER(vectorPointerVariant.at(i)));
       }
     }
   
    private:
     std::vector<boost::variant<VariantTypes...>>& vectorPointerVariant;
   };
   
   #define CEREAL_VECTOR_VARIANT_POINTER(T) cereal::make_vector_pointer_variant(T)
   
   } // namespace cereal
   
   #endif // CEREAL_POINTER_VECTOR_VARIANT_WRAPPER_HPP
   
