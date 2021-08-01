
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_zero_visitor.hpp:

Program Listing for File gradient_zero_visitor.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_gradient_zero_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/gradient_zero_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_GRADIENT_ZERO_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_GRADIENT_ZERO_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   /*
    * GradientZeroVisitor set the gradient to zero for the given module.
    */
   class GradientZeroVisitor : public boost::static_visitor<void>
   {
    public:
     GradientZeroVisitor();
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     template<typename T>
     typename std::enable_if<
         HasGradientCheck<T, arma::mat&(T::*)()>::value, void>::type
     LayerGradients(T* layer, arma::mat& input) const;
   
     template<typename T, typename P>
     typename std::enable_if<
         !HasGradientCheck<T, P&(T::*)()>::value, void>::type
     LayerGradients(T* layer, P& input) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "gradient_zero_visitor_impl.hpp"
   
   #endif
