
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_nystroem_method.hpp:

Program Listing for File nystroem_method.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_nystroem_method_nystroem_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/nystroem_method/nystroem_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_HPP
   #define MLPACK_METHODS_NYSTROEM_METHOD_NYSTROEM_METHOD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "kmeans_selection.hpp"
   
   namespace mlpack {
   namespace kernel {
   
   template<
     typename KernelType,
     typename PointSelectionPolicy = KMeansSelection<>
   >
   class NystroemMethod
   {
    public:
     NystroemMethod(const arma::mat& data, KernelType& kernel, const size_t rank);
   
     void Apply(arma::mat& output);
   
     void GetKernelMatrix(const arma::mat* data,
                          arma::mat& miniKernel,
                          arma::mat& semiKernel);
   
     void GetKernelMatrix(const arma::Col<size_t>& selectedPoints,
                          arma::mat& miniKernel,
                          arma::mat& semiKernel);
   
    private:
     const arma::mat& data;
     KernelType& kernel;
     const size_t rank;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   // Include implementation.
   #include "nystroem_method_impl.hpp"
   
   #endif
