
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_add_visitor_impl.hpp:

Program Listing for File add_visitor_impl.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_add_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/add_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_ADD_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_ADD_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "add_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename... CustomLayers>
   template<typename T>
   inline AddVisitor<CustomLayers...>::AddVisitor(T newLayer) :
       newLayer(std::move(newLayer))
   {
     /* Nothing to do here. */
   }
   
   template<typename... CustomLayers>
   template<typename LayerType>
   inline void AddVisitor<CustomLayers...>::operator()(LayerType* layer) const
   {
     LayerAdd<LayerType>(layer);
   }
   
   template<typename... CustomLayers>
   inline void AddVisitor<CustomLayers...>::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename... CustomLayers>
   template<typename T>
   inline typename std::enable_if<
       HasAddCheck<T, void(T::*)(LayerTypes<CustomLayers...>)>::value, void>::type
   AddVisitor<CustomLayers...>::LayerAdd(T* layer) const
   {
     layer->Add(newLayer);
   }
   
   template<typename... CustomLayers>
   template<typename T>
   inline typename std::enable_if<
       !HasAddCheck<T, void(T::*)(LayerTypes<CustomLayers...>)>::value, void>::type
   AddVisitor<CustomLayers...>::LayerAdd(T* /* layer */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
