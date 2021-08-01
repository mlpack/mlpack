
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_get_param.hpp:

Program Listing for File get_param.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_get_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/get_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_GET_PARAM_HPP
   #define MLPACK_BINDINGS_R_GET_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   void GetParam(util::ParamData& d,
                 const void* /* input */,
                 void* output)
   {
     *((T**) output) = const_cast<T*>(boost::any_cast<T>(&d.value));
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
