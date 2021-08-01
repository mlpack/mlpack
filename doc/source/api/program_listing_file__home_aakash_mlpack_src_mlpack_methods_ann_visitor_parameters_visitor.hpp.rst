
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_parameters_visitor.hpp:

Program Listing for File parameters_visitor.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_parameters_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/parameters_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_PARAMETERS_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_PARAMETERS_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class ParametersVisitor : public boost::static_visitor<void>
   {
    public:
     ParametersVisitor(arma::mat& parameters);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     arma::mat& parameters;
   
     template<typename T, typename P>
     typename std::enable_if<
         !HasParametersCheck<T, P&(T::*)()>::value, void>::type
     LayerParameters(T* layer, P& output) const;
   
     template<typename T, typename P>
     typename std::enable_if<
         HasParametersCheck<T, P&(T::*)()>::value, void>::type
     LayerParameters(T* layer, P& output) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "parameters_visitor_impl.hpp"
   
   #endif
