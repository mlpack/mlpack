
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_kernels_hyperbolic_tangent_kernel.hpp:

Program Listing for File hyperbolic_tangent_kernel.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_kernels_hyperbolic_tangent_kernel.hpp>` (``/home/aakash/mlpack/src/mlpack/core/kernels/hyperbolic_tangent_kernel.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP
   #define MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace kernel {
   
   class HyperbolicTangentKernel
   {
    public:
     HyperbolicTangentKernel() : scale(1.0), offset(0.0)
     { }
   
     HyperbolicTangentKernel(double scale, double offset) :
         scale(scale), offset(offset)
     { }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b)
     {
       return tanh(scale * arma::dot(a, b) + offset);
     }
   
     double Scale() const { return scale; }
     double& Scale() { return scale; }
   
     double Offset() const { return offset; }
     double& Offset() { return offset; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(scale));
       ar(CEREAL_NVP(offset));
     }
   
    private:
     double scale;
     double offset;
   };
   
   } // namespace kernel
   } // namespace mlpack
   
   #endif
