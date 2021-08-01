
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer_norm_impl.hpp:

Program Listing for File layer_norm_impl.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_layer_norm_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/layer_norm_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LAYERNORM_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_LAYERNORM_IMPL_HPP
   
   // In case it is not included.
   #include "layer_norm.hpp"
   
   namespace mlpack {
   namespace ann { 
   template<typename InputDataType, typename OutputDataType>
   LayerNorm<InputDataType, OutputDataType>::LayerNorm() :
       size(0),
       eps(1e-8),
       loading(false)
   {
     // Nothing to do here.
   }
   
   template <typename InputDataType, typename OutputDataType>
   LayerNorm<InputDataType, OutputDataType>::LayerNorm(
       const size_t size, const double eps) :
       size(size),
       eps(eps),
       loading(false)
   {
     weights.set_size(size + size, 1);
   }
   
   template<typename InputDataType, typename OutputDataType>
   void LayerNorm<InputDataType, OutputDataType>::Reset()
   {
     gamma = arma::mat(weights.memptr(), size, 1, false, false);
     beta = arma::mat(weights.memptr() + gamma.n_elem, size, 1, false, false);
   
     if (!loading)
     {
       gamma.fill(1.0);
       beta.fill(0.0);
     }
   
     loading = false;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void LayerNorm<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     mean = arma::mean(input, 0);
     variance = arma::var(input, 1, 0);
   
     // Normalize the input.
     output = input.each_row() - mean;
     inputMean = output;
     output.each_row() /= arma::sqrt(variance + eps);
   
     // Reused in the backward and gradient step.
     normalized = output;
   
     // Scale and shift the output.
     output.each_col() %= gamma;
     output.each_col() += beta;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void LayerNorm<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& input, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
   {
     const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);
   
     // dl / dxhat.
     const arma::mat norm = gy.each_col() % gamma;
   
     // sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
     const arma::mat var = arma::sum(norm % inputMean, 0) %
         arma::pow(stdInv, 3.0) * -0.5;
   
     // dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
     // dl / dmu * 1 / m.
     g = (norm.each_row() % stdInv) + (inputMean.each_row() %
         var * 2 / input.n_rows);
   
     // sum (dl / dxhat * -1 / stdInv) + variance *
     // (sum -2 * (x - mu)) / m.
     g.each_row() += arma::sum(norm.each_row() % -stdInv, 0) / input.n_rows;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void LayerNorm<InputDataType, OutputDataType>::Gradient(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& gradient)
   {
     gradient.set_size(size + size, 1);
   
     // Step 5: dl / dy * xhat.
     gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum(normalized % error, 1);
   
     // Step 6: dl / dy.
     gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) =
         arma::sum(error, 1);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void LayerNorm<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(size));
   
     if (cereal::is_loading<Archive>())
     {
       weights.set_size(size + size, 1);
       loading = true;
     }
   
     ar(CEREAL_NVP(eps));
     ar(CEREAL_NVP(gamma));
     ar(CEREAL_NVP(beta));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
