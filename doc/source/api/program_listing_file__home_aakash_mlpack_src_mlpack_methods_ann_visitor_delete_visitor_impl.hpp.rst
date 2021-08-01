
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delete_visitor_impl.hpp:

Program Listing for File delete_visitor_impl.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delete_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/delete_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_DELETE_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_DELETE_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "delete_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline typename std::enable_if<
       !HasModelCheck<LayerType>::value, void>::type
   DeleteVisitor::operator()(LayerType* layer) const
   {
     if (layer)
       delete layer;
   }
   
   template<typename LayerType>
   inline typename std::enable_if<
       HasModelCheck<LayerType>::value, void>::type
   DeleteVisitor::operator()(LayerType* layer) const
   {
     if (layer)
     {
       for (size_t i = 0; i < layer->Model().size(); ++i)
         boost::apply_visitor(DeleteVisitor(), layer->Model()[i]);
   
       delete layer;
     }
   }
   
   inline void DeleteVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
