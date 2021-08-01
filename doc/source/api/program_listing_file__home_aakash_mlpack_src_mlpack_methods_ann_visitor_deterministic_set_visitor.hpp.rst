
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_deterministic_set_visitor.hpp:

Program Listing for File deterministic_set_visitor.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_deterministic_set_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/deterministic_set_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_DETERMINISTIC_SET_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_DETERMINISTIC_SET_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class DeterministicSetVisitor : public boost::static_visitor<void>
   {
    public:
     DeterministicSetVisitor(const bool deterministic = true);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     const bool deterministic;
   
     template<typename T>
     typename std::enable_if<
         HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
         HasModelCheck<T>::value, void>::type
     LayerDeterministic(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
         HasModelCheck<T>::value, void>::type
     LayerDeterministic(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
         !HasModelCheck<T>::value, void>::type
     LayerDeterministic(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasDeterministicCheck<T, bool&(T::*)(void)>::value &&
         !HasModelCheck<T>::value, void>::type
     LayerDeterministic(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "deterministic_set_visitor_impl.hpp"
   
   #endif
