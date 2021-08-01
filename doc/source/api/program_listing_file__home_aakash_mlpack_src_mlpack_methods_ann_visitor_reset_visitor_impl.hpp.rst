
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_visitor_impl.hpp:

Program Listing for File reset_visitor_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/reset_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "reset_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline void ResetVisitor::operator()(LayerType* layer) const
   {
     ResetParameter(layer);
   }
   
   inline void ResetVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasResetCheck<T, void(T::*)()>::value &&
       !HasModelCheck<T>::value, void>::type
   ResetVisitor::ResetParameter(T* layer) const
   {
     layer->Reset();
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasResetCheck<T, void(T::*)()>::value &&
       HasModelCheck<T>::value, void>::type
   ResetVisitor::ResetParameter(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(ResetVisitor(), layer->Model()[i]);
     }
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasResetCheck<T, void(T::*)()>::value &&
       HasModelCheck<T>::value, void>::type
   ResetVisitor::ResetParameter(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(ResetVisitor(), layer->Model()[i]);
     }
   
     layer->Reset();
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasResetCheck<T, void(T::*)()>::value &&
       !HasModelCheck<T>::value, void>::type
   ResetVisitor::ResetParameter(T* /* layer */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
