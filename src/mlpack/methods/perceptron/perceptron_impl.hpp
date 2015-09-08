/**
 * @file perceptron_impl.hpp
 * @author Udit Saxena
 *
 * Implementation of Perceptron class.
 */
#ifndef __MLPACK_METHODS_PERCEPTRON_PERCEPTRON_IMPL_HPP
#define __MLPACK_METHODS_PERCEPTRON_PERCEPTRON_IMPL_HPP

#include "perceptron.hpp"

namespace mlpack {
namespace perceptron {

/**
 * Constructor - constructs the perceptron. Or rather, builds the weights
 * matrix, which is later used in classification.  It adds a bias input vector
 * of 1 to the input data to take care of the bias weights.
 *
 * @param data Input, training data.
 * @param labels Labels of dataset.
 * @param maxIterations Maximum number of iterations for the perceptron learning
 *      algorithm.
 */
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
  WeightInitializationPolicy WIP;
  WIP.Initialize(weights, biases, data.n_rows, numClasses);

  // Start training.
  Train(data, labels);
}

/**
 * Alternate constructor which copies parameters from an already initiated
 * perceptron.
 *
 * @param other The other initiated Perceptron object from which we copy the
 *      values from.
 * @param data The data on which to train this Perceptron object on.
 * @param instanceWeights Weight vector to use while training. For boosting
 *      purposes.
 * @param labels The labels of data.
 */
template<
    typename LearnPolicy,
    typename WeightInitializationPolicy,
    typename MatType
>
Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Perceptron(
    const Perceptron<>& other,
    const MatType& data,
    const arma::Row<size_t>& labels,
    const arma::rowvec& instanceWeights) :
    maxIterations(other.maxIterations)
{
  // Insert a row of ones at the top of the training data set.
  WeightInitializationPolicy WIP;
  WIP.Initialize(weights, biases, data.n_rows, other.NumClasses());

  Train(data, labels, instanceWeights);
}

/**
 * Classification function. After training, use the weights matrix to classify
 * test, and put the predicted classes in predictedLabels.
 *
 * @param test Testing data or data to classify.
 * @param predictedLabels Vector to store the predicted classes after
 *      classifying test.
 */
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
  arma::uword maxIndex;

  // Could probably be faster if done in batch.
  for (size_t i = 0; i < test.n_cols; i++)
  {
    tempLabelMat = weights.t() * test.col(i) + biases;
    tempLabelMat.max(maxIndex);
    predictedLabels(0, i) = maxIndex;
  }
}

/**
 * Training function.  It trains on trainData using the cost matrix
 * instanceWeights.
 *
 * @param data Data to train on.
 * @param labels Labels of data.
 * @param instanceWeights Cost matrix. Stores the cost of mispredicting
 *      instances.  This is useful for boosting.
 */
template<
    typename LearnPolicy,
    typename WeightInitializationPolicy,
    typename MatType
>
void Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const arma::rowvec& instanceWeights)
{
  size_t j, i = 0;
  bool converged = false;
  size_t tempLabel;
  arma::uword maxIndexRow, maxIndexCol;
  arma::mat tempLabelMat;

  LearnPolicy LP;

  const bool hasWeights = (instanceWeights.n_elem > 0);

  while ((i < maxIterations) && (!converged))
  {
    // This outer loop is for each iteration, and we use the 'converged'
    // variable for noting whether or not convergence has been reached.
    i++;
    converged = true;

    // Now this inner loop is for going through the dataset in each iteration.
    for (j = 0; j < data.n_cols; j++)
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

//! Serialize the perceptron.
template<typename LearnPolicy,
         typename WeightInitializationPolicy,
         typename MatType>
template<typename Archive>
void Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  // For now, do nothing.
}

} // namespace perceptron
} // namespace mlpack

#endif
