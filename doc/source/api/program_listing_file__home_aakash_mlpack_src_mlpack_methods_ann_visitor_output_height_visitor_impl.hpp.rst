
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_height_visitor_impl.hpp:

Program Listing for File output_height_visitor_impl.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_height_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/output_height_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "output_height_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline size_t OutputHeightVisitor::operator()(LayerType* layer) const
   {
     return LayerOutputHeight(layer);
   }
   
   inline size_t OutputHeightVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputHeight<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   OutputHeightVisitor::LayerOutputHeight(T* /* layer */) const
   {
     return 0;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputHeight<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   OutputHeightVisitor::LayerOutputHeight(T* layer) const
   {
     return layer->OutputHeight();
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputHeight<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   OutputHeightVisitor::LayerOutputHeight(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       size_t outputHeight = boost::apply_visitor(OutputHeightVisitor(),
           layer->Model()[layer->Model().size() - 1 - i]);
   
       if (outputHeight != 0)
       {
         return outputHeight;
       }
     }
   
     return 0;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputHeight<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   OutputHeightVisitor::LayerOutputHeight(T* layer) const
   {
     size_t outputHeight = layer->OutputHeight();
   
     if (outputHeight == 0)
     {
       for (size_t i = 0; i < layer->Model().size(); ++i)
       {
         outputHeight = boost::apply_visitor(OutputHeightVisitor(),
             layer->Model()[layer->Model().size() - 1 - i]);
   
         if (outputHeight != 0)
         {
           return outputHeight;
         }
       }
     }
   
     return outputHeight;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
