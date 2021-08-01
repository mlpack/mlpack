
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_wrapper.hpp:

Program Listing for File pointer_wrapper.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cereal_pointer_wrapper.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cereal/pointer_wrapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CEREAL_POINTER_WRAPPER_HPP
   #define MLPACK_CORE_CEREAL_POINTER_WRAPPER_HPP
   
   #include <cereal/archives/binary.hpp>
   #include <cereal/archives/json.hpp>
   #include <cereal/archives/portable_binary.hpp>
   #include <cereal/archives/xml.hpp>
   #include <cereal/types/memory.hpp>
   
   #if __cplusplus <= 201103L && !defined(_MSC_VER)
   namespace std {
   template<typename T, typename... Args>
   std::unique_ptr<T> make_unique(Args&&... args)
   {
     return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
   }
   } // namepace std
   #endif
   
   namespace cereal {
   
   template<class T>
   class PointerWrapper
   {
    public:
     PointerWrapper(T*& pointer)
       : localPointer(pointer)
     {}
   
     template<class Archive>
     void save(Archive& ar, const uint32_t /*version*/) const
     {
       std::unique_ptr<T> smartPointer;
       if (this->localPointer != NULL)
         smartPointer = std::unique_ptr<T>(localPointer);
       ar(CEREAL_NVP(smartPointer));
       localPointer = smartPointer.release();
     }
   
     template<class Archive>
     void load(Archive& ar, const uint32_t /*version*/)
     {
       std::unique_ptr<T> smartPointer;
       ar(CEREAL_NVP(smartPointer));
       localPointer = smartPointer.release();
     }
   
     T*& release() { return localPointer; }
   
    private:
     T*& localPointer;
   };
   
   template<class T>
   inline PointerWrapper<T>
   make_pointer(T*& t)
   {
     return PointerWrapper<T>(t);
   }
   
   #define CEREAL_POINTER(T) cereal::make_pointer(T)
   
   } // namespace cereal
   
   #endif // CEREAL_POINTER_WRAPPER_HPP
