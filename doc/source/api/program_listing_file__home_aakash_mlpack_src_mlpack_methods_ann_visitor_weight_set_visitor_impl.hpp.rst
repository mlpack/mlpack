
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_weight_set_visitor_impl.hpp:

Program Listing for File weight_set_visitor_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_weight_set_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/weight_set_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "weight_set_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline WeightSetVisitor::WeightSetVisitor(arma::mat& weight,
                                             const size_t offset) :
       weight(weight),
       offset(offset)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline size_t WeightSetVisitor::operator()(LayerType* layer) const
   {
     return LayerSize(layer, layer->OutputParameter());
   }
   
   inline size_t WeightSetVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasParametersCheck<T, P&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   WeightSetVisitor::LayerSize(T* /* layer */, P&& /*output */) const
   {
     return 0;
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasParametersCheck<T, P&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   WeightSetVisitor::LayerSize(T* layer, P&& /*output */) const
   {
     size_t modelOffset = 0;
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       modelOffset += boost::apply_visitor(WeightSetVisitor(
           weight, modelOffset + offset), layer->Model()[i]);
     }
   
     return modelOffset;
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       HasParametersCheck<T, P&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   WeightSetVisitor::LayerSize(T* layer, P&& /* output */) const
   {
     layer->Parameters() = arma::mat(weight.memptr() + offset,
         layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);
   
     return layer->Parameters().n_elem;
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       HasParametersCheck<T, P&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   WeightSetVisitor::LayerSize(T* layer, P&& /* output */) const
   {
     layer->Parameters() = arma::mat(weight.memptr() + offset,
         layer->Parameters().n_rows, layer->Parameters().n_cols, false, false);
   
     size_t modelOffset = layer->Parameters().n_elem;
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       modelOffset += boost::apply_visitor(WeightSetVisitor(
           weight, modelOffset + offset), layer->Model()[i]);
     }
   
     return modelOffset;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
