
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_gini_impurity.hpp:

Program Listing for File gini_impurity.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_gini_impurity.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/gini_impurity.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_GINI_INDEX_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_GINI_INDEX_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace tree {
   
   class GiniImpurity
   {
    public:
     static double Evaluate(const arma::Mat<size_t>& counts)
     {
       // We need to sum over the difference between the un-split node and the
       // split nodes.  First we'll calculate the number of elements in each split
       // and total.
       size_t numElem = 0;
       arma::vec splitCounts(counts.n_cols);
       for (size_t i = 0; i < counts.n_cols; ++i)
       {
         splitCounts[i] = arma::accu(counts.col(i));
         numElem += splitCounts[i];
       }
   
       // Corner case: if there are no elements, the impurity is zero.
       if (numElem == 0)
         return 0.0;
   
       arma::Col<size_t> classCounts = arma::sum(counts, 1);
   
       // Calculate the Gini impurity of the un-split node.
       double impurity = 0.0;
       for (size_t i = 0; i < classCounts.n_elem; ++i)
       {
         const double f = ((double) classCounts[i] / (double) numElem);
         impurity += f * (1.0 - f);
       }
   
       // Now calculate the impurity of the split nodes and subtract them from the
       // overall impurity.
       for (size_t i = 0; i < counts.n_cols; ++i)
       {
         if (splitCounts[i] > 0)
         {
           double splitImpurity = 0.0;
           for (size_t j = 0; j < counts.n_rows; ++j)
           {
             const double f = ((double) counts(j, i) / (double) splitCounts[i]);
             splitImpurity += f * (1.0 - f);
           }
   
           impurity -= ((double) splitCounts[i] / (double) numElem) *
               splitImpurity;
         }
       }
   
       return impurity;
     }
   
     static double Range(const size_t numClasses)
     {
       // The best possible case is that only one class exists, which gives a Gini
       // impurity of 0.  The worst possible case is that the classes are evenly
       // distributed, which gives n * (1/n * (1 - 1/n)) = 1 - 1/n.
       return 1.0 - (1.0 / double(numClasses));
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
