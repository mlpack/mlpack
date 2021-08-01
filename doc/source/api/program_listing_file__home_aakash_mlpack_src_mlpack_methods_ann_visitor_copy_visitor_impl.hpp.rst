
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_copy_visitor_impl.hpp:

Program Listing for File copy_visitor_impl.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_copy_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/copy_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_IMPL_HPP
   
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   template <typename... CustomLayers>
   template <typename LayerType>
   inline LayerTypes<CustomLayers...>
   CopyVisitor<CustomLayers...>::operator()(LayerType* layer) const
   {
     return new LayerType(*layer);
   }
   
   template <typename... CustomLayers>
   inline LayerTypes<CustomLayers...>
   CopyVisitor<CustomLayers...>::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
