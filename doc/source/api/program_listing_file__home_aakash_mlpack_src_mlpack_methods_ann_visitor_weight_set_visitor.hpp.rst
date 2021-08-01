
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_weight_set_visitor.hpp:

Program Listing for File weight_set_visitor.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_weight_set_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/weight_set_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_WEIGHT_SET_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class WeightSetVisitor : public boost::static_visitor<size_t>
   {
    public:
     WeightSetVisitor(arma::mat& weight, const size_t offset = 0);
   
     template<typename LayerType>
     size_t operator()(LayerType* layer) const;
   
     size_t operator()(MoreTypes layer) const;
   
    private:
     arma::mat& weight;
   
     const size_t offset;
   
     template<typename T, typename P>
     typename std::enable_if<
         !HasParametersCheck<T, P&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer, P&& input) const;
   
     template<typename T, typename P>
     typename std::enable_if<
         !HasParametersCheck<T, P&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer, P&& input) const;
   
     template<typename T, typename P>
     typename std::enable_if<
         HasParametersCheck<T, P&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer, P&& input) const;
   
     template<typename T, typename P>
     typename std::enable_if<
         HasParametersCheck<T, P&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer, P&& input) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "weight_set_visitor_impl.hpp"
   
   #endif
