
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_is_serializable.hpp:

Program Listing for File is_serializable.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_is_serializable.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/is_serializable.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_IS_SERIALIZABLE_HPP
   #define MLPACK_BINDINGS_MARKDOWN_IS_SERIALIZABLE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   template<typename T>
   bool IsSerializable(
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
   {
     return false;
   }
   
   template<typename T>
   bool IsSerializable(
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0)
   {
     return true;
   }
   
   template<typename T>
   void IsSerializable(util::ParamData& /* data */,
                       const void* /* input */,
                       void* output)
   {
     *((bool*) output) = IsSerializable<typename std::remove_pointer<T>::type>();
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif
