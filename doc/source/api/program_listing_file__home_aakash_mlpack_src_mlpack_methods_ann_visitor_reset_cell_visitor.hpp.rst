
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_cell_visitor.hpp:

Program Listing for File reset_cell_visitor.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reset_cell_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/reset_cell_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_RESET_CELL_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class ResetCellVisitor : public boost::static_visitor<void>
   {
    public:
     ResetCellVisitor(const size_t size);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     size_t size;
   
     template<typename T>
     typename std::enable_if<
         HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
     ResetCell(T* layer) const;
   
     // the Reset() or Model() function.
     template<typename T>
     typename std::enable_if<
         !HasResetCellCheck<T, void(T::*)(const size_t)>::value, void>::type
     ResetCell(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "reset_cell_visitor_impl.hpp"
   
   #endif
