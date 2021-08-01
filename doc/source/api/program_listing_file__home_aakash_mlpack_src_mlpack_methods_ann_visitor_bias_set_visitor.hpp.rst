
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_bias_set_visitor.hpp:

Program Listing for File bias_set_visitor.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_bias_set_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/bias_set_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_BIAS_SET_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_BIAS_SET_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class BiasSetVisitor : public boost::static_visitor<size_t>
   {
    public:
     BiasSetVisitor(arma::mat& weight, const size_t offset = 0);
   
     template<typename LayerType>
     size_t operator()(LayerType* layer) const;
   
     size_t operator()(MoreTypes layer) const;
   
    private:
     arma::mat& weight;
   
     const size_t offset;
   
     template<typename T>
     typename std::enable_if<
         !HasBiasCheck<T, arma::mat&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasBiasCheck<T, arma::mat&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasBiasCheck<T, arma::mat&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasBiasCheck<T, arma::mat&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerSize(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "bias_set_visitor_impl.hpp"
   
   #endif
