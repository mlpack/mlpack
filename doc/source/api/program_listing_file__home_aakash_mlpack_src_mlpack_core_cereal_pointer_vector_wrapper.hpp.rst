
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_vector_wrapper.hpp:

Program Listing for File pointer_vector_wrapper.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_vector_wrapper.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cereal/pointer_vector_wrapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CEREAL_POINTER_VECTOR_WRAPPER_HPP
   #define MLPACK_CORE_CEREAL_POINTER_VECTOR_WRAPPER_HPP
   
   #include <cereal/archives/json.hpp>
   #include <cereal/archives/portable_binary.hpp>
   #include <cereal/archives/xml.hpp>
   #include <cereal/types/vector.hpp>
   
   #include "pointer_wrapper.hpp"
   
   namespace cereal {
   
   template<class T>
   class PointerVectorWrapper
   {
    public:
     PointerVectorWrapper(std::vector<T*>& pointerVec)
       : pointerVector(pointerVec)
     {}
   
     template<class Archive>
     void save(Archive& ar) const
     {
       size_t vecSize = pointerVector.size();
       ar(CEREAL_NVP(vecSize));
       for (size_t i = 0; i < pointerVector.size(); ++i)
       {
         ar(CEREAL_POINTER(pointerVector.at(i)));
       }
     }
   
     template<class Archive>
     void load(Archive& ar)
     {
       size_t vecSize = 0;
       ar(CEREAL_NVP(vecSize));
       pointerVector.resize(vecSize);
       for (size_t i = 0; i < pointerVector.size(); ++i)
       {
         ar(CEREAL_POINTER(pointerVector.at(i)));
       }
     }
   
    private:
     std::vector<T*>& pointerVector;
   };
   
   template<class T>
   inline PointerVectorWrapper<T>
   make_pointer_vector(std::vector<T*>& t)
   {
     return PointerVectorWrapper<T>(t);
   }
   
   #define CEREAL_VECTOR_POINTER(T) cereal::make_pointer_vector(T)
   
   } // namespace cereal
   
   #endif // CEREAL_POINTER_VECTOR_WRAPPER_HPP
