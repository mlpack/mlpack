
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_forward_visitor_impl.hpp:

Program Listing for File forward_visitor_impl.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_forward_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/forward_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "forward_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline ForwardVisitor::ForwardVisitor(const arma::mat& input, arma::mat& output) :
       input(input),
       output(output)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void ForwardVisitor::operator()(LayerType* layer) const
   {
     layer->Forward(input, output);
   }
   
   inline void ForwardVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
