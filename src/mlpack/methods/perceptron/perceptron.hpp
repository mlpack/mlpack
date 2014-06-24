/*
 * @file: perceptron.hpp
 * @author: Udit Saxena
 *
 *
 * Definition of Perceptron
 */

#ifndef _MLPACK_METHODS_PERCEPTRON_HPP
#define _MLPACK_METHODS_PERCEPTRON_HPP

#include <mlpack/core.hpp>
#include "InitializationMethods/zero_init.hpp"
#include "InitializationMethods/random_init.hpp"
#include "LearnPolicy/SimpleWeightUpdate.hpp"


namespace mlpack {
namespace perceptron {

template <typename LearnPolicy = SimpleWeightUpdate, 
          typename WeightInitializationPolicy = ZeroInitialization, 
          typename MatType = arma::mat>
class Perceptron
{
  /*
  This class implements a simple perceptron i.e. a single layer 
  neural network. It converges if the supplied training dataset is 
  linearly separable.

  LearnPolicy: Options of SimpleWeightUpdate and GradientDescent.
  WeightInitializationPolicy: Option of ZeroInitialization and 
                              RandomInitialization.
  */
public:
  /*
  Constructor - Constructs the perceptron. Or rather, builds the weightVectors
  matrix, which is later used in Classification. 
  It adds a bias input vector of 1 to the input data to take care of the bias
  weights.

  @param: data - Input, training data.
  @param: labels - Labels of dataset.
  @param: iterations - maximum number of iterations the perceptron
                       learn algorithm is to be run.
  */
  Perceptron(const MatType& data, const arma::Row<size_t>& labels, int iterations);

  /*
  Classification function. After training, use the weightVectors matrix to 
  classify test, and put the predicted classes in predictedLabels.

  @param: test - testing data or data to classify. 
  @param: predictedLabels - vector to store the predicted classes after
                            classifying test
  */
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);

private:
  
  /* Stores the class labels for the input data*/
  arma::Row<size_t> classLabels;

  /* Stores the weight vectors for each of the input class labels. */
  arma::mat weightVectors;

  /* Stores the training data to be used later on in UpdateWeights.*/
  arma::mat trainData;

  /*
  This function is called by the constructor to update the weightVectors
  matrix. It decreases the weights of the incorrectly classified class while
  increasing the weight of the correct class it should have been classified to.

  @param: rowIndex - index of the row which has been incorrectly predicted.
  @param: labelIndex - index of the vector in trainData.
  @param: vectorIndex - index of the class which should have been predicted.
  */
  // void UpdateWeights(size_t rowIndex, size_t labelIndex, size_t vectorIndex);
};
} // namespace perceptron
} // namespace mlpack

#include "perceptron_impl.cpp"

#endif