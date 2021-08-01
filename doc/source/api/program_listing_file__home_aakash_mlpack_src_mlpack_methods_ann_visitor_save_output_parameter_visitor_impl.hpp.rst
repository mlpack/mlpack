
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_save_output_parameter_visitor_impl.hpp:

Program Listing for File save_output_parameter_visitor_impl.hpp
===============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_save_output_parameter_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/save_output_parameter_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_SAVE_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_SAVE_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "load_output_parameter_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline SaveOutputParameterVisitor::SaveOutputParameterVisitor(
       std::vector<arma::mat>& parameter) : parameter(parameter)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void SaveOutputParameterVisitor::operator()(LayerType* layer) const
   {
     OutputParameter(layer);
   }
   
   inline void SaveOutputParameterVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasModelCheck<T>::value, void>::type
   SaveOutputParameterVisitor::OutputParameter(T* layer) const
   {
     parameter.push_back(layer->OutputParameter());
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasModelCheck<T>::value, void>::type
   SaveOutputParameterVisitor::OutputParameter(T* layer) const
   {
     parameter.push_back(layer->OutputParameter());
   
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(SaveOutputParameterVisitor(parameter),
           layer->Model()[i]);
     }
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
