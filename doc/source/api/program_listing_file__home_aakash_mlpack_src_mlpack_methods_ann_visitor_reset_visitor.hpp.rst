
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_visitor.hpp:

Program Listing for File reset_visitor.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/reset_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class ResetVisitor : public boost::static_visitor<void>
   {
    public:
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     template<typename T>
     typename std::enable_if<
         HasResetCheck<T, void(T::*)()>::value &&
         !HasModelCheck<T>::value, void>::type
     ResetParameter(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasResetCheck<T, void(T::*)()>::value &&
         HasModelCheck<T>::value, void>::type
     ResetParameter(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasResetCheck<T, void(T::*)()>::value &&
         HasModelCheck<T>::value, void>::type
     ResetParameter(T* layer) const;
   
     // the Reset() or Model() function.
     template<typename T>
     typename std::enable_if<
         !HasResetCheck<T, void(T::*)()>::value &&
         !HasModelCheck<T>::value, void>::type
     ResetParameter(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "reset_visitor_impl.hpp"
   
   #endif
