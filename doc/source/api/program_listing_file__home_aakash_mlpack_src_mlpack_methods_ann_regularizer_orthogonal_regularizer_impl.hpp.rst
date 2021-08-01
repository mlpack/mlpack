
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_orthogonal_regularizer_impl.hpp:

Program Listing for File orthogonal_regularizer_impl.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_orthogonal_regularizer_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/regularizer/orthogonal_regularizer_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_IMPL_HPP
   #define MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_IMPL_HPP
   
   // In case it hasn't been included.
   #include "orthogonal_regularizer.hpp"
   
   namespace mlpack {
   namespace ann {
   
   OrthogonalRegularizer::OrthogonalRegularizer(double factor) :
       factor(factor)
   {
     // Nothing to do here.
   }
   
   template<typename MatType>
   void OrthogonalRegularizer::Evaluate(const MatType& weight, MatType& gradient)
   {
     arma::mat grad = arma::zeros(arma::size(weight));
   
     for (size_t i = 0; i < weight.n_rows; ++i)
     {
       for (size_t j = 0; j < weight.n_rows; ++j)
       {
         if (i == j)
         {
           double s =
               arma::as_scalar(
               arma::sign((weight.row(i) * weight.row(i).t()) - 1));
           grad.row(i) += 2 * s * weight.row(i);
         }
         else
         {
           double s = arma::as_scalar(
               arma::sign(weight.row(i) * weight.row(j).t()));
           grad.row(i) += s * weight.row(j);
           grad.row(j) += s * weight.row(i);
         }
       }
     }
   
     gradient += arma::vectorise(grad) * factor;
   }
   
   template<typename Archive>
   void OrthogonalRegularizer::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(factor));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
