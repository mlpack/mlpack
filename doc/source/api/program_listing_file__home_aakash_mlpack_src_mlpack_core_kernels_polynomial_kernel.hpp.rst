
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_polynomial_kernel.hpp:

Program Listing for File polynomial_kernel.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_polynomial_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/polynomial_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_POLYNOMIAL_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class PolynomialKernel
   {
    public:
     PolynomialKernel(const double degree = 2.0, const double offset = 0.0) :
         degree(degree),
         offset(offset)
     { }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b) const
     {
       return pow((arma::dot(a, b) + offset), degree);
     }
   
     const double& Degree() const { return degree; }
     double& Degree() { return degree; }
   
     const double& Offset() const { return offset; }
     double& Offset() { return offset; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(degree));
       ar(CEREAL_NVP(offset));
     }
   
    private:
     double degree;
     double offset;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
