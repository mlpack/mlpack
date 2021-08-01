
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_loss_visitor.hpp:

Program Listing for File loss_visitor.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_loss_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/loss_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_LOSS_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_LOSS_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class LossVisitor : public boost::static_visitor<double>
   {
    public:
     template<typename LayerType>
     double operator()(LayerType* layer) const;
   
     double operator()(MoreTypes layer) const;
   
    private:
     template<typename T>
     typename std::enable_if<
         !HasLoss<T, double(T::*)()>::value &&
         !HasModelCheck<T>::value, double>::type
     LayerLoss(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasLoss<T, double(T::*)()>::value &&
         !HasModelCheck<T>::value, double>::type
     LayerLoss(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasLoss<T, double(T::*)()>::value &&
         HasModelCheck<T>::value, double>::type
     LayerLoss(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasLoss<T, double(T::*)()>::value &&
         HasModelCheck<T>::value, double>::type
     LayerLoss(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "loss_visitor_impl.hpp"
   
   #endif
