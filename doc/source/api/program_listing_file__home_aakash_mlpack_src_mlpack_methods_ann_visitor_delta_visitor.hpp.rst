
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delta_visitor.hpp:

Program Listing for File delta_visitor.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delta_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/delta_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_DELTA_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_DELTA_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class DeltaVisitor : public boost::static_visitor<arma::mat&>
   {
    public:
     template<typename LayerType>
     arma::mat& operator()(LayerType* layer) const;
   
     arma::mat& operator()(MoreTypes layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "delta_visitor_impl.hpp"
   
   #endif
