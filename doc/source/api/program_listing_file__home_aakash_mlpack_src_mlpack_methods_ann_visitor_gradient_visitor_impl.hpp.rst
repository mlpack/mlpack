
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_visitor_impl.hpp:

Program Listing for File gradient_visitor_impl.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/gradient_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_GRADIENT_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "gradient_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline GradientVisitor::GradientVisitor(const arma::mat& input,
                                           const arma::mat& delta) :
       input(input),
       delta(delta),
       index(0),
       hasIndex(false)
   {
     /* Nothing to do here. */
   }
   
   inline GradientVisitor::GradientVisitor(const arma::mat& input,
                                           const arma::mat& delta,
                                           const size_t index) :
       input(input),
       delta(delta),
       index(index),
       hasIndex(true)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void GradientVisitor::operator()(LayerType* layer) const
   {
     LayerGradients(layer, layer->OutputParameter());
   }
   
   inline void GradientVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasGradientCheck<T, arma::mat&(T::*)()>::value &&
       !HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
   GradientVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
   {
     layer->Gradient(input, delta, layer->Gradient());
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasGradientCheck<T, arma::mat&(T::*)()>::value &&
       HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
   GradientVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
   {
     if (!hasIndex)
     {
       layer->Gradient(input, delta, layer->Gradient());
     }
     else
     {
       layer->Gradient(input, delta, layer->Gradient(), index);
     }
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasGradientCheck<T, P&(T::*)()>::value, void>::type
   GradientVisitor::LayerGradients(T* /* layer */, P& /* input */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
