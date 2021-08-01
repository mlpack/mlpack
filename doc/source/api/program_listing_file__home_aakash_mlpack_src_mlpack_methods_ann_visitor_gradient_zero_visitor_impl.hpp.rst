
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_zero_visitor_impl.hpp:

Program Listing for File gradient_zero_visitor_impl.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_zero_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/gradient_zero_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_ZERO_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_GRADIENT_ZERO_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "gradient_zero_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline GradientZeroVisitor::GradientZeroVisitor()
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void GradientZeroVisitor::operator()(LayerType* layer) const
   {
     LayerGradients(layer, layer->OutputParameter());
   }
   
   inline void GradientZeroVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasGradientCheck<T, arma::mat&(T::*)()>::value, void>::type
   GradientZeroVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
   {
     layer->Gradient().zeros();
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasGradientCheck<T, P&(T::*)()>::value, void>::type
   GradientZeroVisitor::LayerGradients(T* /* layer */, P& /* input */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
