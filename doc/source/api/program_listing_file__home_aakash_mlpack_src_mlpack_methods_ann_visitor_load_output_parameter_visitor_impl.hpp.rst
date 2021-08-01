
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_load_output_parameter_visitor_impl.hpp:

Program Listing for File load_output_parameter_visitor_impl.hpp
===============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_load_output_parameter_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/load_output_parameter_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_LOAD_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_LOAD_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "load_output_parameter_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline LoadOutputParameterVisitor::LoadOutputParameterVisitor(
       std::vector<arma::mat>& parameter) : parameter(parameter)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void LoadOutputParameterVisitor::operator()(LayerType* layer) const
   {
     OutputParameter(layer);
   }
   
   inline void LoadOutputParameterVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasModelCheck<T>::value, void>::type
   LoadOutputParameterVisitor::OutputParameter(T* layer) const
   {
     layer->OutputParameter() = parameter.back();
     parameter.pop_back();
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasModelCheck<T>::value, void>::type
   LoadOutputParameterVisitor::OutputParameter(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(LoadOutputParameterVisitor(parameter),
           layer->Model()[layer->Model().size() - i - 1]);
     }
   
     layer->OutputParameter() = parameter.back();
     parameter.pop_back();
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
