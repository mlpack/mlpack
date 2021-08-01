
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_tests_get_param.hpp:

Program Listing for File get_param.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_tests_get_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/tests/get_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_TESTS_GET_PARAM_HPP
   #define MLPACK_BINDINGS_TESTS_GET_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace tests {
   
   template<typename T>
   T& GetParam(util::ParamData& d)
   {
     // No mapping is needed, so just cast it directly.
     return *boost::any_cast<T>(&d.value);
   }
   
   template<typename T>
   void GetParam(util::ParamData& d, const void* /* input */, void* output)
   {
     // Cast to the correct type.
     *((T**) output) = &GetParam<T>(const_cast<util::ParamData&>(d));
   }
   
   } // namespace tests
   } // namespace bindings
   } // namespace mlpack
   
   #endif
