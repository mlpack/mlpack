
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_decision_tree_gini_gain.hpp:

Program Listing for File gini_gain.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_decision_tree_gini_gain.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/decision_tree/gini_gain.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DECISION_TREE_GINI_GAIN_HPP
   #define MLPACK_METHODS_DECISION_TREE_GINI_GAIN_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace tree {
   
   class GiniGain
   {
    public:
     template<bool UseWeights, typename CountType>
     static double EvaluatePtr(const CountType* counts,
                               const size_t countLength,
                               const CountType totalCount)
     {
       if (totalCount == 0)
         return 0.0;
   
       CountType impurity = 0.0;
       for (size_t i = 0; i < countLength; ++i)
         impurity += counts[i] * (totalCount - counts[i]);
   
       return -((double) impurity / ((double) std::pow(totalCount, 2)));
     }
   
     template<bool UseWeights, typename RowType, typename WeightVecType>
     static double Evaluate(const RowType& labels,
                            const size_t numClasses,
                            const WeightVecType& weights)
     {
       // Corner case: if there are no elements, the impurity is zero.
       if (labels.n_elem == 0)
         return 0.0;
   
       // Count the number of elements in each class.  Use four auxiliary vectors
       // to exploit SIMD instructions if possible.
       arma::vec countSpace(4 * numClasses, arma::fill::zeros);
       arma::vec counts(countSpace.memptr(), numClasses, false, true);
       arma::vec counts2(countSpace.memptr() + numClasses, numClasses, false,
           true);
       arma::vec counts3(countSpace.memptr() + 2 * numClasses, numClasses, false,
           true);
       arma::vec counts4(countSpace.memptr() + 3 * numClasses, numClasses, false,
           true);
   
       // Calculate the Gini impurity of the un-split node.
       double impurity = 0.0;
   
       if (UseWeights)
       {
         // Sum all the weights up.
         double accWeights[4] = { 0.0, 0.0, 0.0, 0.0 };
   
         // SIMD loop: add counts for four elements simultaneously (if the compiler
         // manages to vectorize the loop).
         for (size_t i = 3; i < labels.n_elem; i += 4)
         {
           const double weight1 = weights[i - 3];
           const double weight2 = weights[i - 2];
           const double weight3 = weights[i - 1];
           const double weight4 = weights[i];
   
           counts[labels[i - 3]] += weight1;
           counts2[labels[i - 2]] += weight2;
           counts3[labels[i - 1]] += weight3;
           counts4[labels[i]] += weight4;
   
           accWeights[0] += weight1;
           accWeights[1] += weight2;
           accWeights[2] += weight3;
           accWeights[3] += weight4;
         }
   
         // Handle leftovers.
         if (labels.n_elem % 4 == 1)
         {
           const double weight1 = weights[labels.n_elem - 1];
           counts[labels[labels.n_elem - 1]] += weight1;
           accWeights[0] += weight1;
         }
         else if (labels.n_elem % 4 == 2)
         {
           const double weight1 = weights[labels.n_elem - 2];
           const double weight2 = weights[labels.n_elem - 1];
   
           counts[labels[labels.n_elem - 2]] += weight1;
           counts2[labels[labels.n_elem - 1]] += weight2;
   
           accWeights[0] += weight1;
           accWeights[1] += weight2;
         }
         else if (labels.n_elem % 4 == 3)
         {
           const double weight1 = weights[labels.n_elem - 3];
           const double weight2 = weights[labels.n_elem - 2];
           const double weight3 = weights[labels.n_elem - 1];
   
           counts[labels[labels.n_elem - 3]] += weight1;
           counts2[labels[labels.n_elem - 2]] += weight2;
           counts3[labels[labels.n_elem - 1]] += weight3;
   
           accWeights[0] += weight1;
           accWeights[1] += weight2;
           accWeights[2] += weight3;
         }
   
         accWeights[0] += accWeights[1] + accWeights[2] + accWeights[3];
         counts += counts2 + counts3 + counts4;
   
         // Catch edge case: if there are no weights, the impurity is zero.
         if (accWeights[0] == 0.0)
           return 0.0;
   
         for (size_t i = 0; i < numClasses; ++i)
         {
           const double f = ((double) counts[i] / (double) accWeights[0]);
           impurity += f * (1.0 - f);
         }
       }
       else
       {
         // SIMD loop: add counts for four elements simultaneously (if the compiler
         // manages to vectorize the loop).
         for (size_t i = 3; i < labels.n_elem; i += 4)
         {
           counts[labels[i - 3]]++;
           counts2[labels[i - 2]]++;
           counts3[labels[i - 1]]++;
           counts4[labels[i]]++;
         }
   
         // Handle leftovers.
         if (labels.n_elem % 4 == 1)
         {
           counts[labels[labels.n_elem - 1]]++;
         }
         else if (labels.n_elem % 4 == 2)
         {
           counts[labels[labels.n_elem - 2]]++;
           counts2[labels[labels.n_elem - 1]]++;
         }
         else if (labels.n_elem % 4 == 3)
         {
           counts[labels[labels.n_elem - 3]]++;
           counts2[labels[labels.n_elem - 2]]++;
           counts3[labels[labels.n_elem - 1]]++;
         }
   
         counts += counts2 + counts3 + counts4;
   
         for (size_t i = 0; i < numClasses; ++i)
         {
           const double f = ((double) counts[i] / (double) labels.n_elem);
           impurity += f * (1.0 - f);
         }
       }
   
       return -impurity;
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
