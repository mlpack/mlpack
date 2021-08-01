
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_multiple_random_dimension_select.hpp:

Program Listing for File multiple_random_dimension_select.hpp
=============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_multiple_random_dimension_select.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/multiple_random_dimension_select.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP
   #define MLPACK_METHODS_DECISION_TREE_MULTIPLE_RANDOM_DIMENSION_SPLIT_HPP
   
   namespace mlpack {
   namespace tree {
   
   class MultipleRandomDimensionSelect
   {
    public:
     MultipleRandomDimensionSelect(const size_t numDimensions = 0) :
           numDimensions(numDimensions),
           i(0),
           dimensions(0)
     { }
   
     size_t Begin()
     {
       // Reset if possible.
       if (numDimensions == 0 || numDimensions > dimensions)
         numDimensions = (size_t) std::sqrt(dimensions);
   
       values.set_size(numDimensions + 1);
   
       // Try setting new values.
       for (size_t i = 0; i < numDimensions; ++i)
       {
         // Generate random different numbers.
         bool unique = false;
         size_t value;
         while (!unique)
         {
           value = math::RandInt(dimensions);
   
           // Check if we already have the value.
           unique = true;
           for (size_t j = 0; j < i; ++j)
           {
             if (values[j] == value)
             {
               unique = false;
               break;
             }
           }
         }
   
         values[i] = value;
       }
   
       values[numDimensions] = std::numeric_limits<size_t>::max();
   
       i = 0;
       return values[0];
     }
   
     size_t End() const { return size_t(-1); }
   
     size_t Next()
     {
       return values[++i];
     }
   
     size_t Dimensions() const { return dimensions; }
     size_t& Dimensions() { return dimensions; }
   
    private:
     size_t numDimensions;
     arma::Col<size_t> values;
     size_t i;
     size_t dimensions;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
