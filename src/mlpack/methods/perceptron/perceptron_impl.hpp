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
 * Constructor - constructs the perceptron. Or rather, builds the weightVectors
 * matrix, which is later used in Classification.
 * It adds a bias input vector of 1 to the input data to take care of the bias
 * weights.
 *
 * @param data Input, training data.
 * @param labels Labels of dataset.
 * @param iterations Maximum number of iterations for the perceptron learning
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
    int iterations)
{
  WeightInitializationPolicy WIP;
  WIP.Initialize(weightVectors, arma::max(labels) + 1, data.n_rows + 1);

  // Start training.
  classLabels = labels;

  trainData = data;
  // Insert a row of ones at the top of the training data set.
  MatType zOnes(1, data.n_cols);
  zOnes.fill(1);
  trainData.insert_rows(0, zOnes);

  int j, i = 0;
  bool converged = false;
  size_t tempLabel;
  arma::uword maxIndexRow, maxIndexCol;
  arma::mat tempLabelMat;

  LearnPolicy LP;

  while ((i < iterations) && (!converged))
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
      tempLabelMat = weightVectors * trainData.col(j);

      tempLabelMat.max(maxIndexRow, maxIndexCol);
      
      // Check whether prediction is correct.
      if (maxIndexRow != classLabels(0, j))
      {
        // Due to incorrect prediction, convergence set to false.
        converged = false;
        tempLabel = labels(0, j);
        // Send maxIndexRow for knowing which weight to update, send j to know
        // the value of the vector to update it with.  Send tempLabel to know
        // the correct class.
        LP.UpdateWeights(trainData, weightVectors, j, tempLabel, maxIndexRow);
      }
    }
  }
}

/**
 * Classification function. After training, use the weightVectors matrix to
 * classify test, and put the predicted classes in predictedLabels.
 *
 * @param test testing data or data to classify.
 * @param predictedLabels vector to store the predicted classes after
 *      classifying test
 */
template <typename LearnPolicy, typename WeightInitializationPolicy, typename MatType>
void Perceptron<LearnPolicy, WeightInitializationPolicy, MatType>::Classify(
    const MatType& test,
    arma::Row<size_t>& predictedLabels)
{
  arma::mat tempLabelMat;
  arma::uword maxIndexRow, maxIndexCol;

  for (int i = 0; i < test.n_cols; i++)
  {
    tempLabelMat = weightVectors.submat(0, 1, weightVectors.n_rows - 1,
                                        weightVectors.n_cols - 1) * 
                                        test.col(i) + weightVectors.col(0);
    tempLabelMat.max(maxIndexRow, maxIndexCol);
    predictedLabels(0, i) = maxIndexRow;
  }
}

}; // namespace perceptron
}; // namespace mlpack

#endif
