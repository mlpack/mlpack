
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_parameter_visitor_impl.hpp:

Program Listing for File output_parameter_visitor_impl.hpp
==========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_parameter_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/output_parameter_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "output_parameter_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<typename LayerType>
   inline arma::mat& OutputParameterVisitor::operator()(LayerType *layer) const
   {
     return layer->OutputParameter();
   }
   
   inline arma::mat& OutputParameterVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
