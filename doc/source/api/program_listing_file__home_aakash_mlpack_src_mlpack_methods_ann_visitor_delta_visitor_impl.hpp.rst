
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delta_visitor_impl.hpp:

Program Listing for File delta_visitor_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delta_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/delta_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_DELTA_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_DELTA_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "delta_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline arma::mat& DeltaVisitor::operator()(LayerType *layer) const
   {
     return layer->Delta();
   }
   
   inline arma::mat& DeltaVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
