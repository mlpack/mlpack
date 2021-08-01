
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_backward_visitor_impl.hpp:

Program Listing for File backward_visitor_impl.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_backward_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/backward_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "backward_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline BackwardVisitor::BackwardVisitor(const arma::mat& input,
                                           const arma::mat& error,
                                           arma::mat& delta) :
     input(input),
     error(error),
     delta(delta),
     index(0),
     hasIndex(false)
   {
     /* Nothing to do here. */
   }
   
   inline BackwardVisitor::BackwardVisitor(const arma::mat& input,
                                           const arma::mat& error,
                                           arma::mat& delta,
                                           const size_t index) :
     input(input),
     error(error),
     delta(delta),
     index(index),
     hasIndex(true)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void BackwardVisitor::operator()(LayerType* layer) const
   {
     LayerBackward(layer, layer->OutputParameter());
   }
   
   inline void BackwardVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
   BackwardVisitor::LayerBackward(T* layer, arma::mat& /* input */) const
   {
     layer->Backward(input, error, delta);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
   BackwardVisitor::LayerBackward(T* layer, arma::mat& /* input */) const
   {
     if (!hasIndex)
     {
       layer->Backward(input, error, delta);
     }
     else
     {
       layer->Backward(input, error, delta, index);
     }
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
