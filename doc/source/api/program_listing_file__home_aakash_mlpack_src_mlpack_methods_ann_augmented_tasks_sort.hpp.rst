
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_sort.hpp:

Program Listing for File sort.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_sort.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/augmented/tasks/sort.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AUGMENTED_TASKS_SORT_HPP
   #define MLPACK_METHODS_AUGMENTED_TASKS_SORT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann /* Artificial Neural Network */ {
   namespace augmented /* Augmented neural network */ {
   namespace tasks /* Task utilities for augmented */ {
   
   class SortTask
   {
    public:
     SortTask(const size_t maxLength,
              const size_t bitLen,
              bool addSeparator = false);
   
     void Generate(arma::field<arma::mat>& input,
                   arma::field<arma::mat>& labels,
                         const size_t batchSize,
                         bool fixedLength = false) const;
     void Generate(arma::mat& input,
                   arma::mat& labels,
                   const size_t batchSize) const;
   
    private:
     // Maximum length of the sequence.
     size_t maxLength;
     // Binary length of sorted numbers.
     size_t bitLen;
     // Flag indicating whether generator should produce
     // separator as part of the sequence
     bool addSeparator;
   };
   
   } // namespace tasks
   } // namespace augmented
   } // namespace ann
   } // namespace mlpack
   
   #include "sort_impl.hpp"
   #endif
