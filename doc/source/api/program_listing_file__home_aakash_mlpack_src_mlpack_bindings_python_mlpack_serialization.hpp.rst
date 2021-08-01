
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_mlpack_serialization.hpp:

Program Listing for File serialization.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_mlpack_serialization.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/mlpack/serialization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_MLPACK_SERIALIZATION_HPP
   #define MLPACK_BINDINGS_PYTHON_MLPACK_SERIALIZATION_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   std::string SerializeOut(T* t, const std::string& name)
   {
     std::ostringstream oss;
     {
       cereal::BinaryOutputArchive b(oss);
   
       b(cereal::make_nvp(name.c_str(), *t));
     }
     return oss.str();
   }
   
   template<typename T>
   void SerializeIn(T* t, const std::string& str, const std::string& name)
   {
     std::istringstream iss(str);
     cereal::BinaryInputArchive b(iss);
     b(cereal::make_nvp(name.c_str(), *t));
   }
   
   template<typename T>
   std::string SerializeOutJSON(T* t, const std::string& name)
   {
     std::ostringstream oss;
     {
       cereal::JSONOutputArchive b(oss);
   
       b(cereal::make_nvp(name.c_str(), *t));
     }
     return oss.str();
   }
   
   template<typename T>
   void SerializeInJSON(T* t, const std::string& str, const std::string& name)
   {
     std::istringstream iss(str);
     cereal::JSONInputArchive b(iss);
     b(cereal::make_nvp(name.c_str(), *t));
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
