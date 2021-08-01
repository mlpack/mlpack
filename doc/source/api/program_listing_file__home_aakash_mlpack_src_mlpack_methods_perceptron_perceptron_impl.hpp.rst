
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_perceptron_perceptron_impl.hpp:

Program Listing for File perceptron_impl.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_perceptron_perceptron_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/perceptron/perceptron_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PERCEPTRON_PERCEPTRON_IMPL_HPP
   #define MLPACK_METHODS_PERCEPTRON_PERCEPTRON_IMPL_HPP
   
   #include "perceptron.hpp"
   
   namespace mlpack {
   namespace perceptron {
   
   template<
       typename LearnPolicy,
       typename WeightInitializationPolicy,
       typename MatType
   >
   Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Perceptron(
       const size_t numClasses,
       const size_t dimensionality,
       const size_t maxIterations) :
       maxIterations(maxIterations)
   {
     WeightInitializationPolicy wip;
     wip.Initialize(weights, biases, dimensionality, numClasses);
   }
   
   template<
       typename LearnPolicy,
       typename WeightInitializationPolicy,
       typename MatType
   >
   Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Perceptron(
       const MatType& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses,
       const size_t maxIterations) :
       maxIterations(maxIterations)
   {
     // Start training.
     Train(data, labels, numClasses);
   }
   
   template<
       typename LearnPolicy,
       typename WeightInitializationPolicy,
       typename MatType
   >
   Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Perceptron(
       const Perceptron& other,
       const MatType& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses,
       const arma::rowvec& instanceWeights) :
       maxIterations(other.maxIterations)
   {
     Train(data, labels, numClasses, instanceWeights);
   }
   
   template<
       typename LearnPolicy,
       typename WeightInitializationPolicy,
       typename MatType
   >
   void Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Classify(
       const MatType& test,
       arma::Row<size_t>& predictedLabels)
   {
     arma::vec tempLabelMat;
     arma::uword maxIndex = 0;
   
     // Could probably be faster if done in batch.
     for (size_t i = 0; i < test.n_cols; ++i)
     {
       tempLabelMat = weights.t() * test.col(i) + biases;
       tempLabelMat.max(maxIndex);
       predictedLabels(0, i) = maxIndex;
     }
   }
   
   template<
       typename LearnPolicy,
       typename WeightInitializationPolicy,
       typename MatType
   >
   void Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Train(
       const MatType& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses,
       const arma::rowvec& instanceWeights)
   {
     // Do we need to resize the weights?
     if (weights.n_elem != numClasses)
     {
       WeightInitializationPolicy wip;
       wip.Initialize(weights, biases, data.n_rows, numClasses);
     }
   
     size_t j, i = 0;
     bool converged = false;
     size_t tempLabel;
     arma::uword maxIndexRow = 0, maxIndexCol = 0;
     arma::mat tempLabelMat;
   
     LearnPolicy LP;
   
     const bool hasWeights = (instanceWeights.n_elem > 0);
   
     while ((i < maxIterations) && (!converged))
     {
       // This outer loop is for each iteration, and we use the 'converged'
       // variable for noting whether or not convergence has been reached.
       ++i;
       converged = true;
   
       // Now this inner loop is for going through the dataset in each iteration.
       for (j = 0; j < data.n_cols; ++j)
       {
         // Multiply for each variable and check whether the current weight vector
         // correctly classifies this.
         tempLabelMat = weights.t() * data.col(j) + biases;
   
         tempLabelMat.max(maxIndexRow, maxIndexCol);
   
         // Check whether prediction is correct.
         if (maxIndexRow != labels(0, j))
         {
           // Due to incorrect prediction, convergence set to false.
           converged = false;
           tempLabel = labels(0, j);
   
           // Send maxIndexRow for knowing which weight to update, send j to know
           // the value of the vector to update it with.  Send tempLabel to know
           // the correct class.
           if (hasWeights)
             LP.UpdateWeights(data.col(j), weights, biases, maxIndexRow, tempLabel,
                 instanceWeights(j));
           else
             LP.UpdateWeights(data.col(j), weights, biases, maxIndexRow,
                 tempLabel);
         }
       }
     }
   }
   
   template<typename LearnPolicy,
            typename WeightInitializationPolicy,
            typename MatType>
   template<typename Archive>
   void Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     // We just need to serialize the maximum number of iterations, the weights,
     // and the biases.
     ar(CEREAL_NVP(maxIterations));
     ar(CEREAL_NVP(weights));
     ar(CEREAL_NVP(biases));
   }
   
   } // namespace perceptron
   } // namespace mlpack
   
   #endif
