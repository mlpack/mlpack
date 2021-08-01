
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_delete_allocated_memory.hpp:

Program Listing for File delete_allocated_memory.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_delete_allocated_memory.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/delete_allocated_memory.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_DELETE_ALLOCATED_MEMORY_HPP
   #define MLPACK_BINDINGS_CLI_DELETE_ALLOCATED_MEMORY_HPP
   
   #include <mlpack/core/util/param_data.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   void DeleteAllocatedMemoryImpl(
       util::ParamData& /* d */,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0)
   {
     // Do nothing.
   }
   
   template<typename T>
   void DeleteAllocatedMemoryImpl(
       util::ParamData& /* d */,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     // Do nothing.
   }
   
   template<typename T>
   void DeleteAllocatedMemoryImpl(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Delete the allocated memory (hopefully we actually own it).
     typedef std::tuple<T*, std::string> TupleType;
     delete std::get<0>(*boost::any_cast<TupleType>(&d.value));
   }
   
   template<typename T>
   void DeleteAllocatedMemory(
       util::ParamData& d,
       const void* /* input */,
       void* /* output */)
   {
     DeleteAllocatedMemoryImpl<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
