
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_random_forest_bootstrap.hpp:

Program Listing for File bootstrap.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_random_forest_bootstrap.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/random_forest/bootstrap.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANDOM_FOREST_BOOTSTRAP_HPP
   #define MLPACK_METHODS_RANDOM_FOREST_BOOTSTRAP_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<bool UseWeights,
            typename MatType,
            typename LabelsType,
            typename WeightsType>
   void Bootstrap(const MatType& dataset,
                  const LabelsType& labels,
                  const WeightsType& weights,
                  MatType& bootstrapDataset,
                  LabelsType& bootstrapLabels,
                  WeightsType& bootstrapWeights)
   {
     bootstrapDataset.set_size(dataset.n_rows, dataset.n_cols);
     bootstrapLabels.set_size(labels.n_elem);
     if (UseWeights)
       bootstrapWeights.set_size(weights.n_elem);
   
     // Random sampling with replacement.
     arma::uvec indices = arma::randi<arma::uvec>(dataset.n_cols,
         arma::distr_param(0, dataset.n_cols - 1));
     bootstrapDataset = dataset.cols(indices);
     bootstrapLabels = labels.cols(indices);
     if (UseWeights)
       bootstrapWeights = weights.cols(indices);
   }
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
