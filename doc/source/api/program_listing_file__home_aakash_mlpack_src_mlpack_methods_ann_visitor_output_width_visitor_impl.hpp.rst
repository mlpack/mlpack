
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_width_visitor_impl.hpp:

Program Listing for File output_width_visitor_impl.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_width_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/output_width_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_WIDTH_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_OUTPUT_WIDTH_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "output_width_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline size_t OutputWidthVisitor::operator()(LayerType* layer) const
   {
     return LayerOutputWidth(layer);
   }
   
   inline size_t OutputWidthVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputWidth<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   OutputWidthVisitor::LayerOutputWidth(T* /* layer */) const
   {
     return 0;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputWidth<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   OutputWidthVisitor::LayerOutputWidth(T* layer) const
   {
     return layer->OutputWidth();
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputWidth<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   OutputWidthVisitor::LayerOutputWidth(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       size_t outputWidth = boost::apply_visitor(OutputWidthVisitor(),
           layer->Model()[layer->Model().size() - 1 - i]);
   
       if (outputWidth != 0)
       {
         return outputWidth;
       }
     }
   
     return 0;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputWidth<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   OutputWidthVisitor::LayerOutputWidth(T* layer) const
   {
     size_t outputWidth = layer->OutputWidth();
   
     if (outputWidth == 0)
     {
       for (size_t i = 0; i < layer->Model().size(); ++i)
       {
         outputWidth = boost::apply_visitor(OutputWidthVisitor(),
             layer->Model()[layer->Model().size() - 1 - i]);
   
         if (outputWidth != 0)
         {
           return outputWidth;
         }
       }
     }
   
     return outputWidth;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
