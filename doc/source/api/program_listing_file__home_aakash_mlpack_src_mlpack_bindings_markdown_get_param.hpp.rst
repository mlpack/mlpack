
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_param.hpp:

Program Listing for File get_param.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/get_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_GET_PARAM_HPP
   #define MLPACK_BINDINGS_MARKDOWN_GET_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   template<typename T>
   void GetParam(util::ParamData& d,
                 const void* /* input */,
                 void* output)
   {
     util::ParamData& dmod = const_cast<util::ParamData&>(d);
     *((T**) output) = boost::any_cast<T>(&dmod.value);
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif
