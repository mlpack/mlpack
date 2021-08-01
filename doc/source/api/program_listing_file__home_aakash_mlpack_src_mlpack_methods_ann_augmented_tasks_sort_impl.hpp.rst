
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_sort_impl.hpp:

Program Listing for File sort_impl.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_sort_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/augmented/tasks/sort_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AUGMENTED_TASKS_SORT_IMPL_HPP
   #define MLPACK_METHODS_AUGMENTED_TASKS_SORT_IMPL_HPP
   
   #include "sort.hpp"
   
   namespace mlpack {
   namespace ann /* Artificial Neural Network */ {
   namespace augmented /* Augmented neural network */ {
   namespace tasks /* Task utilities for augmented */ {
   
   SortTask::SortTask(const size_t maxLength,
                      const size_t bitLen,
                      bool addSeparator)
     : maxLength(maxLength), bitLen(bitLen), addSeparator(addSeparator)
   {
     if (maxLength <= 1)
     {
       std::ostringstream oss;
       oss << "SortTask::SortTask(): maximum sequence length ("
           << maxLength << ") "
           << "should be at least 2!"
           << std::endl;
       throw std::invalid_argument(oss.str());
     }
     if (bitLen <= 0)
     {
       std::ostringstream oss;
       oss << "SortTask::SortTask(): binary length (" << bitLen << ") "
           << "is not positive!"
           << std::endl;
       throw std::invalid_argument(oss.str());
     }
   }
   
   void SortTask::Generate(arma::field<arma::mat>& input,
                           arma::field<arma::mat>& labels,
                           const size_t batchSize,
                           bool fixedLength) const
   {
     input = arma::field<arma::mat>(batchSize);
     labels = arma::field<arma::mat>(batchSize);
     size_t size = maxLength;
     for (size_t i = 0; i < batchSize; ++i)
     {
       if (!fixedLength)
       {
         // Generate random uniform length from [2..maxLength].
         size = mlpack::math::RandInt(2, maxLength+1);
       }
       input(i) = arma::randi<arma::mat>(bitLen, size, arma::distr_param(0, 1));
       arma::mat itemAns(bitLen, size);
       arma::colvec vals(size);
       for (size_t j = 0; j < size; ++j)
       {
         int val = 0;
         for (size_t k = 0; k < bitLen; ++k)
         {
           val <<= 1;
           val += input(i).at(k, j);
         }
         vals[j] = val;
       }
       arma::uvec indices = arma::sort_index(vals);
       for (size_t j = 0; j < size; ++j)
       {
         itemAns.col(j) = input(i).col(indices.at(j));
       }
       labels(i) = itemAns;
       input(i).reshape(input(i).n_elem, 1);
       if (addSeparator)
       {
         arma::mat sepInput = arma::zeros(input(i).n_elem + size, 1);
         size_t ptr = 0, origPtr = 0;
         for (size_t j = 0; j < size; ++j)
         {
           sepInput.rows(ptr, ptr + bitLen - 1) =
             input(i).rows(origPtr, origPtr + bitLen - 1);
           ptr += bitLen;
           origPtr += bitLen;
           sepInput.at(ptr, 0) = 0.5;
           ++ptr;
         }
         input(i) = sepInput;
       }
       labels(i).reshape(labels(i).n_elem, 1);
     }
   }
   
   void SortTask::Generate(arma::mat& input,
                           arma::mat& labels,
                           const size_t batchSize) const
   {
     arma::field<arma::mat> fieldInput, fieldLabels;
     Generate(fieldInput, fieldLabels, batchSize, true);
     size_t inputRows = fieldInput(0).n_rows;
     size_t labelRows = fieldLabels(0).n_rows;
     size_t cols = batchSize;
     input = arma::zeros(inputRows, cols);
     labels = arma::zeros(labelRows, cols);
     for (size_t i = 0; i < cols; ++i)
     {
       input.col(i) = fieldInput.at(i);
       labels.col(i) = fieldLabels.at(i);
     }
   }
   
   } // namespace tasks
   } // namespace augmented
   } // namespace ann
   } // namespace mlpack
   #endif
