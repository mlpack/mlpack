
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delete_visitor.hpp:

Program Listing for File delete_visitor.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_delete_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/delete_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_DELETE_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_DELETE_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class DeleteVisitor : public boost::static_visitor<void>
   {
    public:
     template<typename LayerType>
     typename std::enable_if<
         !HasModelCheck<LayerType>::value, void>::type
     operator()(LayerType* layer) const;
   
     template<typename LayerType>
     typename std::enable_if<
         HasModelCheck<LayerType>::value, void>::type
     operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "delete_visitor_impl.hpp"
   
   #endif
