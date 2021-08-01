
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_set_input_height_visitor_impl.hpp:

Program Listing for File set_input_height_visitor_impl.hpp
==========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_set_input_height_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/set_input_height_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_HEIGHT_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_HEIGHT_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "set_input_height_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline SetInputHeightVisitor::SetInputHeightVisitor(const size_t inputHeight,
                                                       const bool reset) :
       inputHeight(inputHeight),
       reset(reset)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline bool SetInputHeightVisitor::operator()(LayerType* layer) const
   {
     return LayerInputHeight(layer);
   }
   
   inline bool SetInputHeightVisitor::operator()(MoreTypes layer) const
   {
     return layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputHeight<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, bool>::type
   SetInputHeightVisitor::LayerInputHeight(T* /* layer */) const
   {
     return false;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputHeight<T, size_t&(T::*)()>::value &&
       !HasModelCheck<T>::value, bool>::type
   SetInputHeightVisitor::LayerInputHeight(T* layer) const
   {
     if (layer->InputHeight() == 0 || reset)
     {
       layer->InputHeight() = inputHeight;
     }
   
     return true;
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasInputHeight<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, bool>::type
   SetInputHeightVisitor::LayerInputHeight(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(SetInputHeightVisitor(inputHeight, reset),
           layer->Model()[i]);
     }
   
     return true;
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasInputHeight<T, size_t&(T::*)()>::value &&
       HasModelCheck<T>::value, bool>::type
   SetInputHeightVisitor::LayerInputHeight(T* layer) const
   {
     if (layer->InputHeight() == 0  || reset)
     {
       layer->InputHeight() = inputHeight;
     }
   
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(SetInputHeightVisitor(inputHeight, reset),
           layer->Model()[i]);
     }
   
     return true;
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
