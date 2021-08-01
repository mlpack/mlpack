
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_radial_basis_function.hpp:

Program Listing for File radial_basis_function.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_radial_basis_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/radial_basis_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_RBF_HPP
   #define MLPACK_METHODS_ANN_LAYER_RBF_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/activation_functions/gaussian_function.hpp>
   
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat,
       typename Activation = GaussianFunction
   >
   class RBF
   {
    public:
     RBF();
   
     RBF(const size_t inSize,
         const size_t outSize,
         arma::mat& centres,
         double betas = 0);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& /* gy */,
                   arma::Mat<eT>& /* g */);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     InputDataType const& InputParameter() const { return inputParameter; }
     InputDataType& InputParameter() { return inputParameter; }
   
     size_t InputSize() const { return inSize; }
   
     size_t OutputSize() const { return outSize; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t WeightSize() const
     {
       return 0;
     }
   
     size_t InputShape() const
     {
       return inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inSize;
   
     size_t outSize;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     double sigmas;
   
     double betas;
   
     InputDataType centres;
   
     InputDataType inputParameter;
   
     OutputDataType distances;
   }; // class RBF
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "radial_basis_function_impl.hpp"
   
   #endif
