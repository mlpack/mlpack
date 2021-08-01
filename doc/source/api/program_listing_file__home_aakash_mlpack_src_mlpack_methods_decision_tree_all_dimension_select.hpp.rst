
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_all_dimension_select.hpp:

Program Listing for File all_dimension_select.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_all_dimension_select.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/all_dimension_select.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_ALL_DIMENSION_SELECT_HPP
   #define MLPACK_METHODS_DECISION_TREE_ALL_DIMENSION_SELECT_HPP
   
   namespace mlpack {
   namespace tree {
   
   class AllDimensionSelect
   {
    public:
     AllDimensionSelect() : i(0), dimensions(0) { }
   
     size_t Begin()
     {
       i = 0;
       return 0;
     }
   
     size_t End() const { return dimensions; }
   
     size_t Next() { return ++i; }
   
     size_t Dimensions() const { return dimensions; }
     size_t& Dimensions() { return dimensions; }
   
    private:
     size_t i;
     size_t dimensions;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
