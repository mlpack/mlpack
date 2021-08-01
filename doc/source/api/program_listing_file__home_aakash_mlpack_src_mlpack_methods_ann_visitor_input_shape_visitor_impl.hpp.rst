
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_input_shape_visitor_impl.hpp:

Program Listing for File input_shape_visitor_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_input_shape_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/input_shape_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "input_shape_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline std::size_t InShapeVisitor::operator()(LayerType* layer) const
   {
     return LayerInputShape(layer);
   }
   
   inline std::size_t InShapeVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputShapeCheck<T>::value, std::size_t>::type
   InShapeVisitor::LayerInputShape(T* /* layer */) const
   {
     return 0;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputShapeCheck<T>::value, std::size_t>::type
   InShapeVisitor::LayerInputShape(T* layer) const
   {
     return layer->InputShape();
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
