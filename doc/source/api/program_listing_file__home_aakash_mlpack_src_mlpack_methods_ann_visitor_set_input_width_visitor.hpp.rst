
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_set_input_width_visitor.hpp:

Program Listing for File set_input_width_visitor.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_set_input_width_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/set_input_width_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_SET_INPUT_WIDTH_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class SetInputWidthVisitor : public boost::static_visitor<bool>
   {
    public:
     SetInputWidthVisitor(const size_t inputWidth = 0, const bool reset = false);
   
     template<typename LayerType>
     bool operator()(LayerType* layer) const;
   
     bool operator()(MoreTypes layer) const;
   
    private:
     size_t inputWidth;
   
     bool reset;
   
     template<typename T>
     typename std::enable_if<
         !HasInputWidth<T, size_t&(T::*)()>::value &&
         !HasModelCheck<T>::value, bool>::type
     LayerInputWidth(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasInputWidth<T, size_t&(T::*)()>::value &&
         !HasModelCheck<T>::value, bool>::type
     LayerInputWidth(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasInputWidth<T, size_t&(T::*)()>::value &&
         HasModelCheck<T>::value, bool>::type
     LayerInputWidth(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasInputWidth<T, size_t&(T::*)()>::value &&
         HasModelCheck<T>::value, bool>::type
     LayerInputWidth(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "set_input_width_visitor_impl.hpp"
   
   #endif
