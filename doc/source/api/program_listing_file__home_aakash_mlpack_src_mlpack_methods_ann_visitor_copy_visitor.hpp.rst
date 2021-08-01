
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_copy_visitor.hpp:

Program Listing for File copy_visitor.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_copy_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/copy_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   template <typename... CustomLayers>
   class CopyVisitor : public boost::static_visitor<LayerTypes<CustomLayers...> >
   {
    public:
     template <typename LayerType>
     LayerTypes<CustomLayers...> operator()(LayerType*) const;
   
     LayerTypes<CustomLayers...> operator()(MoreTypes) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation
   #include "copy_visitor_impl.hpp"
   #endif
   
