
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_input_shape_visitor.hpp:

Program Listing for File input_shape_visitor.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_input_shape_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/input_shape_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_INPUT_SHAPE_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class InShapeVisitor : public boost::static_visitor<size_t>
   {
    public:
     template<typename LayerType>
     size_t operator()(LayerType* layer) const;
   
     size_t operator()(MoreTypes layer) const;
   
    private:
     template<typename T>
     typename std::enable_if<
         !HasInputShapeCheck<T>::value, size_t>::type
     LayerInputShape(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasInputShapeCheck<T>::value, size_t>::type
     LayerInputShape(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "input_shape_visitor_impl.hpp"
   
   #endif
