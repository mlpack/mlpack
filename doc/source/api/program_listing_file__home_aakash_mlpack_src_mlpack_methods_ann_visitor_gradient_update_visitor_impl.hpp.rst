
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_update_visitor_impl.hpp:

Program Listing for File gradient_update_visitor_impl.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_update_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/gradient_update_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_UPDATE_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_GRADIENT_UPDATE_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "gradient_update_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline GradientUpdateVisitor::GradientUpdateVisitor(arma::mat& gradient,
                                                       size_t offset) :
       gradient(gradient),
       offset(offset)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline size_t GradientUpdateVisitor::operator()(LayerType* layer) const
   {
     return LayerGradients(layer, layer->OutputParameter());
   }
   
   inline size_t GradientUpdateVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasGradientCheck<T, arma::mat&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   GradientUpdateVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
   {
     if (layer->Parameters().n_elem != 0)
     {
       layer->Gradient() = gradient.submat(offset, 0,
           offset + layer->Parameters().n_elem - 1, 0);;
     }
   
     return layer->Parameters().n_elem;
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   GradientUpdateVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
   {
     size_t modelOffset = 0;
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       modelOffset += boost::apply_visitor(GradientUpdateVisitor(
           gradient, modelOffset + offset), layer->Model()[i]);
     }
   
     return modelOffset;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasGradientCheck<T, arma::mat&(T::*)()>::value &&
       HasModelCheck<T>::value, size_t>::type
   GradientUpdateVisitor::LayerGradients(T* layer, arma::mat& /* input */) const
   {
     if (layer->Parameters().n_elem != 0)
     {
       layer->Gradient() = gradient.submat(offset, 0,
           offset + layer->Parameters().n_elem - 1, 0);;
     }
   
     size_t modelOffset = layer->Parameters().n_elem;
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       modelOffset += boost::apply_visitor(GradientUpdateVisitor(
           gradient, modelOffset + offset), layer->Model()[i]);
     }
   
     return modelOffset;
   }
   
   template<typename T, typename P>
   inline typename std::enable_if<
       !HasGradientCheck<T, P&(T::*)()>::value &&
       !HasModelCheck<T>::value, size_t>::type
   GradientUpdateVisitor::LayerGradients(T* /* layer */, P& /* input */) const
   {
     return 0;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
