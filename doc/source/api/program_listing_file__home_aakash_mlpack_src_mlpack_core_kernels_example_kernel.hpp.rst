
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_example_kernel.hpp:

Program Listing for File example_kernel.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_example_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/example_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_EXAMPLE_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_EXAMPLE_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   
   namespace kernel {
   
   class ExampleKernel
   {
    public:
     ExampleKernel() { }
   
     template<typename VecTypeA, typename VecTypeB>
     static double Evaluate(const VecTypeA& /* a */, const VecTypeB& /* b */)
     { return 0; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   
     template<typename VecTypeA, typename VecTypeB>
     static double ConvolutionIntegral(const VecTypeA& /* a */,
                                       const VecTypeB& /* b */) { return 0; }
   
     static double Normalizer() { return 0; }
   
     // Modified to remove unused variable "dimension"
     // static double Normalizer(size_t dimension=1) { return 0; }
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
