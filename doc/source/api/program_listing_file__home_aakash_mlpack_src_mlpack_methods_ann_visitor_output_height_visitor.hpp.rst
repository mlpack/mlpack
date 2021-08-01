
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_height_visitor.hpp:

Program Listing for File output_height_visitor.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_output_height_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/output_height_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_OUTPUT_HEIGHT_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class OutputHeightVisitor : public boost::static_visitor<size_t>
   {
    public:
     template<typename LayerType>
     size_t operator()(LayerType* layer) const;
   
     size_t operator()(MoreTypes layer) const;
   
    private:
     template<typename T>
     typename std::enable_if<
         !HasInputHeight<T, size_t&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerOutputHeight(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasInputHeight<T, size_t&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerOutputHeight(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasInputHeight<T, size_t&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerOutputHeight(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasInputHeight<T, size_t&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerOutputHeight(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "output_height_visitor_impl.hpp"
   
   #endif
