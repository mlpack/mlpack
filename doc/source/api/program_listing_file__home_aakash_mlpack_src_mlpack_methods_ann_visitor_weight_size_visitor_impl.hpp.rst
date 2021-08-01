
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_weight_size_visitor_impl.hpp:

Program Listing for File weight_size_visitor_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_weight_size_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/weight_size_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_WEIGHT_SIZE_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_WEIGHT_SIZE_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "weight_size_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline size_t WeightSizeVisitor::operator()(LayerType* layer) const
   {
     return LayerSize(layer, layer->OutputParameter());
   }
   
   inline size_t WeightSizeVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasParametersCheck<T, P&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   WeightSizeVisitor::LayerSize(T* /* layer */, P& /* output */) const
   {
     return 0;
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasParametersCheck<T, P&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
   {
     size_t weights = 0;
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       weights += boost::apply_visitor(WeightSizeVisitor(), layer->Model()[i]);
     }
   
     return weights;
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       HasParametersCheck<T, P&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
   {
     return layer->Parameters().n_elem;
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       HasParametersCheck<T, P&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   WeightSizeVisitor::LayerSize(T* layer, P& /* output */) const
   {
     size_t weights = layer->Parameters().n_elem;
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       weights += boost::apply_visitor(WeightSizeVisitor(), layer->Model()[i]);
     }
   
     return weights;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
