
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_add.hpp:

Program Listing for File add.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_add.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/augmented/tasks/add.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AUGMENTED_TASKS_ADD_HPP
   #define MLPACK_METHODS_AUGMENTED_TASKS_ADD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/discrete_distribution.hpp>
   
   namespace mlpack {
   namespace ann /* Artificial Neural Network */ {
   namespace augmented /* Augmented neural network */ {
   namespace tasks /* Task utilities for augmented */ {
   
   class AddTask
   {
    public:
     AddTask(const size_t bitLen);
   
     void Generate(arma::field<arma::mat>& input,
                   arma::field<arma::mat>& labels,
                   const size_t batchSize,
                   const bool fixedLength = false) const;
   
     void Generate(arma::mat& input,
                   arma::mat& labels,
                   const size_t batchSize) const;
   
    private:
     // Maximum binary length of numbers.
     size_t bitLen;
   
     void Binarize(const arma::field<arma::vec>& input,
                   arma::field<arma::mat>& output) const;
   };
   
   } // namespace tasks
   } // namespace augmented
   } // namespace ann
   } // namespace mlpack
   
   #include "add_impl.hpp"
   #endif
