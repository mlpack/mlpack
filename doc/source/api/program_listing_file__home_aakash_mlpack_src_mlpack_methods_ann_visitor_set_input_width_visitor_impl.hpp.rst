
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_set_input_width_visitor_impl.hpp:

Program Listing for File set_input_width_visitor_impl.hpp
=========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_set_input_width_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/set_input_width_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "set_input_width_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline SetInputWidthVisitor::SetInputWidthVisitor(const size_t inputWidth,
                                                     const bool reset) :
       inputWidth(inputWidth),
       reset(reset)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline bool SetInputWidthVisitor::operator()(LayerType* layer) const
   {
     return LayerInputWidth(layer);
   }
   
   inline bool SetInputWidthVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputWidth<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, bool>::type
   SetInputWidthVisitor::LayerInputWidth(T* /* layer */) const
   {
     return false;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputWidth<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, bool>::type
   SetInputWidthVisitor::LayerInputWidth(T* layer) const
   {
     if (layer->InputWidth() == 0 || reset)
     {
       layer->InputWidth() = inputWidth;
     }
   
     return true;
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputWidth<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, bool>::type
   SetInputWidthVisitor::LayerInputWidth(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(SetInputWidthVisitor(inputWidth, reset),
           layer->Model()[i]);
     }
   
     return true;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputWidth<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, bool>::type
   SetInputWidthVisitor::LayerInputWidth(T* layer) const
   {
     if (layer->InputWidth() == 0 || reset)
     {
       layer->InputWidth() = inputWidth;
     }
   
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(SetInputWidthVisitor(inputWidth, reset),
           layer->Model()[i]);
     }
   
     return true;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
