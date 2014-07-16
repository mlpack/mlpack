/**
 * @file perceptron.hpp
 * @author Udit Saxena
 *
 * Definition of Perceptron class.
 */
#ifndef __MLPACK_METHODS_PERCEPTRON_PERCEPTRON_HPP
#define __MLPACK_METHODS_PERCEPTRON_PERCEPTRON_HPP

#include <mlpack/core.hpp>

#include "initialization_methods/zero_init.hpp"
#include "initialization_methods/random_init.hpp"
#include "learning_policies/simple_weight_update.hpp"

namespace mlpack {
namespace perceptron {

/**
 * This class implements a simple perceptron (i.e., a single layer neural
 * network).  It converges if the supplied training dataset is linearly
 * separable.
 *
 * @tparam LearnPolicy Options of SimpleWeightUpdate and GradientDescent.
 * @tparam WeightInitializationPolicy Option of ZeroInitialization and
 *      RandomInitialization.
 */
template<typename LearnPolicy = SimpleWeightUpdate,
         typename WeightInitializationPolicy = ZeroInitialization,
         typename MatType = arma::mat>
class Perceptron
{
 public:
  /**
   * Constructor - constructs the perceptron by building the weightVectors
   * matrix, which is later used in Classification.  It adds a bias input vector
   * of 1 to the input data to take care of the bias weights.
   *
   * @param data Input, training data.
   * @param labels Labels of dataset.
   * @param iterations Maximum number of iterations for the perceptron learning
   *     algorithm.
   */
  Perceptron(const MatType& data, const arma::Row<size_t>& labels, int iterations);

  /**
   * Classification function. After training, use the weightVectors matrix to
   * classify test, and put the predicted classes in predictedLabels.
   *
   * @param test Testing data or data to classify.
   * @param predictedLabels Vector to store the predicted classes after
   *     classifying test.
   */
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);

  /**
   *
   *
   *
   */
  Perceptron(const Perceptron<>& other, MatType& data, const arma::Row<double>& D, const arma::Row<size_t>& labels);

private:
  //! To store the number of iterations
  size_t iter;

  //! Stores the class labels for the input data.
  arma::Row<size_t> classLabels;

  //! Stores the weight vectors for each of the input class labels.
  arma::mat weightVectors;

  //! Stores the training data to be used later on in UpdateWeights.
  arma::mat trainData;

  /**
   * Train function.
   *
   */
  void Train();
};

} // namespace perceptron
} // namespace mlpack

#include "perceptron_impl.hpp"

#endif
