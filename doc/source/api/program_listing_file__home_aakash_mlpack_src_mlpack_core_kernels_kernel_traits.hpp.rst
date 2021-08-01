
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_kernel_traits.hpp:

Program Listing for File kernel_traits.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_kernel_traits.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/kernel_traits.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_KERNEL_TRAITS_HPP
   #define MLPACK_CORE_KERNELS_KERNEL_TRAITS_HPP
   
   namespace mlpack {
   namespace kernel {
   
   template<typename KernelType>
   class KernelTraits
   {
    public:
     static const bool IsNormalized = false;
   
     static const bool UsesSquaredDistance = false;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
