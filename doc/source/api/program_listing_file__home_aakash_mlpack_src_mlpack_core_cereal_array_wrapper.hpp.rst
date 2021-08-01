
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_array_wrapper.hpp:

Program Listing for File array_wrapper.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cereal_array_wrapper.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cereal/array_wrapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CEREAL_ARRAY_WRAPPER_HPP
   #define MLPACK_CORE_CEREAL_ARRAY_WRAPPER_HPP
   
   #include <cereal/archives/binary.hpp>
   #include <cereal/archives/portable_binary.hpp>
   #include <cereal/archives/xml.hpp>
   #include <cereal/archives/json.hpp>
   
   namespace cereal {
   
   template<class T>
   class ArrayWrapper
   {
    public:
     ArrayWrapper(T*& addr, std::size_t& size) :
         arrayAddress(addr),
         arraySize(size)
     {}
   
     template<class Archive>
     void save(Archive& ar) const
     {
       ar(CEREAL_NVP(arraySize));
       for (size_t i = 0; i < arraySize; ++i)
         ar(cereal::make_nvp("item", arrayAddress[i]));
     }
   
     template<class Archive>
     void load(Archive& ar)
     {
       ar(CEREAL_NVP(arraySize));
       delete[] arrayAddress;
       if (arraySize == 0)
       {
         arrayAddress = NULL;
         return;
       }
       arrayAddress = new T[arraySize];
       for (size_t i = 0; i < arraySize; ++i)
         ar(cereal::make_nvp("item", arrayAddress[i]));
     }
   
    private:
     ArrayWrapper& operator=(ArrayWrapper rhs);
   
     T*& arrayAddress;
     size_t& arraySize;
   };
   
   template<class T, class S>
   inline
   ArrayWrapper<T> make_array(T*& t, S& s)
   {
     return ArrayWrapper<T>(t, s);
   }
   
   #define CEREAL_POINTER_ARRAY(T, S) cereal::make_array(T, S)
   
   } // namespace cereal
   
   #endif // CEREAL_ARRAY_WRAPPER_HPP
