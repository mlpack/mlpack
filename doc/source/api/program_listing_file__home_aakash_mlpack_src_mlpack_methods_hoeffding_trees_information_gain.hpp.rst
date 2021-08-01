
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_information_gain.hpp:

Program Listing for File information_gain.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hoeffding_trees_information_gain.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hoeffding_trees/information_gain.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HOEFFDING_TREES_INFORMATION_GAIN_HPP
   #define MLPACK_METHODS_HOEFFDING_TREES_INFORMATION_GAIN_HPP
   
   namespace mlpack {
   namespace tree {
   
   class HoeffdingInformationGain
   {
    public:
     static double Evaluate(const arma::Mat<size_t>& counts)
     {
       // Calculate the number of elements in the unsplit node and also in each
       // proposed child.
       size_t numElem = 0;
       arma::vec splitCounts(counts.n_elem);
       for (size_t i = 0; i < counts.n_cols; ++i)
       {
         splitCounts[i] = arma::accu(counts.col(i));
         numElem += splitCounts[i];
       }
   
       // Corner case: if there are no elements, the gain is zero.
       if (numElem == 0)
         return 0.0;
   
       arma::Col<size_t> classCounts = arma::sum(counts, 1);
   
       // Calculate the gain of the unsplit node.
       double gain = 0.0;
       for (size_t i = 0; i < classCounts.n_elem; ++i)
       {
         const double f = ((double) classCounts[i] / (double) numElem);
         if (f > 0.0)
           gain -= f * std::log2(f);
       }
   
       // Now calculate the impurity of the split nodes and subtract them from the
       // overall gain.
       for (size_t i = 0; i < counts.n_cols; ++i)
       {
         if (splitCounts[i] > 0)
         {
           double splitGain = 0.0;
           for (size_t j = 0; j < counts.n_rows; ++j)
           {
             const double f = ((double) counts(j, i) / (double) splitCounts[i]);
             if (f > 0.0)
               splitGain += f * std::log2(f);
           }
   
           gain += ((double) splitCounts[i] / (double) numElem) * splitGain;
         }
       }
   
       return gain;
     }
   
     static double Range(const size_t numClasses)
     {
       // The best possible case gives an information gain of 0.  The worst
       // possible case is even distribution, which gives n * (1/n * log2(1/n)) =
       // log2(1/n) = -log2(n).  So, the range is log2(n).
       return std::log2(numClasses);
     }
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
