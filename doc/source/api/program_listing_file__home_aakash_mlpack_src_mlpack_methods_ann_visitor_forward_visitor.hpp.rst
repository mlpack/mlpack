
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_forward_visitor.hpp:

Program Listing for File forward_visitor.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_forward_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/forward_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_FORWARD_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class ForwardVisitor : public boost::static_visitor<void>
   {
    public:
     ForwardVisitor(const arma::mat& input, arma::mat& output);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     const arma::mat& input;
   
     arma::mat& output;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "forward_visitor_impl.hpp"
   
   #endif
