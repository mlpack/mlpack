
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_linear_kernel.hpp:

Program Listing for File linear_kernel.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_linear_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/linear_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_LINEAR_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_LINEAR_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class LinearKernel
   {
    public:
     LinearKernel() { }
   
     template<typename VecTypeA, typename VecTypeB>
     static double Evaluate(const VecTypeA& a, const VecTypeB& b)
     {
       return arma::dot(a, b);
     }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */) { }
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
