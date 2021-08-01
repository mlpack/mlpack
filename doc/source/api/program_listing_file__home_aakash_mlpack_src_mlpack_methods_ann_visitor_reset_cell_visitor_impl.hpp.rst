
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_cell_visitor_impl.hpp:

Program Listing for File reset_cell_visitor_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_cell_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/reset_cell_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "reset_cell_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline ResetCellVisitor::ResetCellVisitor(const size_t size) : size(size)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void ResetCellVisitor::operator()(LayerType* layer) const
   {
     ResetCell(layer);
   }
   
   inline void ResetCellVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
   ResetCellVisitor::ResetCell(T* layer) const
   {
     layer->ResetCell(size);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
   ResetCellVisitor::ResetCell(T* /* layer */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
