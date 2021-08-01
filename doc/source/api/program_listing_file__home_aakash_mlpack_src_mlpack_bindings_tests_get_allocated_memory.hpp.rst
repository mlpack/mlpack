
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_tests_get_allocated_memory.hpp:

Program Listing for File get_allocated_memory.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_tests_get_allocated_memory.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/tests/get_allocated_memory.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_IO_GET_ALLOCATED_MEMORY_HPP
   #define MLPACK_BINDINGS_IO_GET_ALLOCATED_MEMORY_HPP
   
   #include <mlpack/core/util/param_data.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace tests {
   
   template<typename T>
   void* GetAllocatedMemory(
       util::ParamData& /* d */,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0)
   {
     return NULL;
   }
   
   template<typename T>
   void* GetAllocatedMemory(
       util::ParamData& /* d */,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     return NULL;
   }
   
   template<typename T>
   void* GetAllocatedMemory(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Here we have a model; return its memory location.
     return *boost::any_cast<T*>(&d.value);
   }
   
   template<typename T>
   void GetAllocatedMemory(util::ParamData& d,
                           const void* /* input */,
                           void* output)
   {
     *((void**) output) =
         GetAllocatedMemory<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace tests
   } // namespace bindings
   } // namespace mlpack
   
   #endif
