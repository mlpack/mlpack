
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_parameters_set_visitor_impl.hpp:

Program Listing for File parameters_set_visitor_impl.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_parameters_set_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/parameters_set_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_PARAMETERS_SET_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_PARAMETERS_SET_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "parameters_set_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline ParametersSetVisitor::ParametersSetVisitor(arma::mat& parameters) :
       parameters(parameters)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void ParametersSetVisitor::operator()(LayerType *layer) const
   {
     LayerParameters(layer, layer->OutputParameter());
   }
   
   inline void ParametersSetVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasParametersCheck<T, P&(T::*)()>::value, void>::type
   ParametersSetVisitor::LayerParameters(T* /* layer */, P& /* output */) const
   {
     /* Nothing to do here. */
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       HasParametersCheck<T, P&(T::*)()>::value, void>::type
   ParametersSetVisitor::LayerParameters(T* layer, P& /* output */) const
   {
     layer->Parameters() = parameters;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
