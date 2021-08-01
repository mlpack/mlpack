
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_deterministic_set_visitor_impl.hpp:

Program Listing for File deterministic_set_visitor_impl.hpp
===========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_deterministic_set_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/deterministic_set_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_DETERMINISTIC_SET_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_DETERMINISTIC_SET_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "deterministic_set_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline DeterministicSetVisitor::DeterministicSetVisitor(
       const bool deterministic) : deterministic(deterministic)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void DeterministicSetVisitor::operator()(LayerType* layer) const
   {
     LayerDeterministic(layer);
   }
   
   inline void DeterministicSetVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
       HasModelCheck<T>::value, void>::type
   DeterministicSetVisitor::LayerDeterministic(T* layer) const
   {
     layer->Deterministic() = deterministic;
   
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(DeterministicSetVisitor(deterministic),
           layer->Model()[i]);
     }
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
       HasModelCheck<T>::value, void>::type
   DeterministicSetVisitor::LayerDeterministic(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(DeterministicSetVisitor(deterministic),
           layer->Model()[i]);
     }
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
       !HasModelCheck<T>::value, void>::type
   DeterministicSetVisitor::LayerDeterministic(T* layer) const
   {
     layer->Deterministic() = deterministic;
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
       !HasModelCheck<T>::value, void>::type
   DeterministicSetVisitor::LayerDeterministic(T* /* input */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
