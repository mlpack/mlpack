
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_update_visitor.hpp:

Program Listing for File gradient_update_visitor.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_update_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/gradient_update_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_UPDATE_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_GRADIENT_UPDATE_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class GradientUpdateVisitor : public boost::static_visitor<size_t>
   {
    public:
     GradientUpdateVisitor(arma::mat& gradient, size_t offset = 0);
   
     template<typename LayerType>
     size_t operator()(LayerType* layer) const;
   
     size_t operator()(MoreTypes layer) const;
   
    private:
     arma::mat& gradient;
   
     size_t offset;
   
     template<typename T>
     typename std::enable_if<
         HasGradientCheck<T, arma::mat&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerGradients(T* layer, arma::mat& input) const;
   
     template<typename T>
     typename std::enable_if<
         !HasGradientCheck<T, arma::mat&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerGradients(T* layer, arma::mat& input) const;
   
     template<typename T>
     typename std::enable_if<
         HasGradientCheck<T, arma::mat&(T::*)()>::value &&
         HasModelCheck<T>::value, size_t>::type
     LayerGradients(T* layer, arma::mat& input) const;
   
     template<typename T, typename P>
     typename std::enable_if<
         !HasGradientCheck<T, P&(T::*)()>::value &&
         !HasModelCheck<T>::value, size_t>::type
     LayerGradients(T* layer, P& input) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "gradient_update_visitor_impl.hpp"
   
   #endif
