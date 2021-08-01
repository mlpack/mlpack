
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_score.hpp:

Program Listing for File score.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_augmented_tasks_score.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/augmented/tasks/score.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_AUGMENTED_TASKS_SCORE_HPP
   #define MLPACK_METHODS_AUGMENTED_TASKS_SCORE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann /* Artificial Neural Network */ {
   namespace augmented /* Augmented neural network */ {
   namespace scorers /* Scoring utilities for augmented */ {
   
   template<typename MatType>
   double SequencePrecision(arma::field<MatType> trueOutputs,
                            arma::field<MatType> predOutputs,
                            double tol = 1e-4);
   
   } // namespace scorers
   } // namespace augmented
   } // namespace ann
   } // namespace mlpack
   
   #include "score_impl.hpp"
   
   #endif
