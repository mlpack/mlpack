/**
 * @file perceptron.hpp
 * @author Udit Saxena
 *
 * Definition of Perceptron class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_PERCEPTRON_PERCEPTRON_HPP
#define MLPACK_METHODS_PERCEPTRON_PERCEPTRON_HPP

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
   * Constructor: create the perceptron with the given number of classes and
   * initialize the weight matrix, but do not perform any training.  (Call the
   * Train() function to perform training.)
   *
   * @param numClasses Number of classes in the dataset.
   * @param dimensionality Dimensionality of the dataset.
   * @param maxIterations Maximum number of iterations for the perceptron
   *      learning algorithm.
   */
  Perceptron(const size_t numClasses = 0,
             const size_t dimensionality = 0,
             const size_t maxIterations = 1000);

  /**
   * Constructor: constructs the perceptron by building the weights matrix,
   * which is later used in classification.  The number of classes should be
   * specified separately, and the labels vector should contain values in the
   * range [0, numClasses - 1].  The data::NormalizeLabels() function can be
   * used if the labels vector does not contain values in the required range.
   *
   * @param data Input, training data.
   * @param labels Labels of dataset.
   * @param numClasses Number of classes in the dataset.
   * @param maxIterations Maximum number of iterations for the perceptron
   *      learning algorithm.
   */
  Perceptron(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t maxIterations = 1000);

  /**
   * Alternate constructor which copies parameters from an already initiated
   * perceptron.
   *
   * @param other The other initiated Perceptron object from which we copy the
   *       values from.
   * @param data The data on which to train this Perceptron object on.
   * @param D Weight vector to use while training. For boosting purposes.
   * @param labels The labels of data.
   */
  Perceptron(const Perceptron<>& other,
             const MatType& data,
             const arma::Row<size_t>& labels,
             const arma::rowvec& instanceWeights);

  /**
   * Train the perceptron on the given data for up to the maximum number of
   * iterations (specified in the constructor or through MaxIterations()).  A
   * single iteration corresponds to a single pass through the data, so if you
   * want to pass through the dataset only once, set MaxIterations() to 1.
   *
   * This training does not reset the model weights, so you can call Train() on
   * multiple datasets sequentially.
   *
   * @param data Dataset on which training should be performed.
   * @param labels Labels of the dataset.  Make sure that these labels don't
   *      contain any values greater than NumClasses()!
   * @param instanceWeights Cost matrix. Stores the cost of mispredicting
   *      instances.  This is useful for boosting.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const arma::rowvec& instanceWeights = arma::rowvec());

  /**
   * Classification function. After training, use the weights matrix to
   * classify test, and put the predicted classes in predictedLabels.
   *
   * @param test Testing data or data to classify.
   * @param predictedLabels Vector to store the predicted classes after
   *     classifying test.
   */
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);

  /**
   * Serialize the perceptron.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the number of classes this perceptron has been trained for.
  size_t NumClasses() const { return weights.n_cols; }

  //! Get the weight matrix.
  const arma::mat& Weights() const { return weights; }
  //! Modify the weight matrix.  You had better know what you are doing!
  arma::mat& Weights() { return weights; }

  //! Get the biases.
  const arma::vec& Biases() const { return biases; }
  //! Modify the biases.  You had better know what you are doing!
  arma::vec& Biases() { return biases; }

private:
  //! The maximum number of iterations during training.
  size_t maxIterations;

  /**
   * Stores the weights for each of the input class labels.  Each column
   * corresponds to the weights for one class label, and each row corresponds to
   * the weights for one dimension of the input data.  The biases are held in a
   * separate vector.
   */
  arma::mat weights;

  //! The biases for each class.
  arma::vec biases;
};

} // namespace perceptron
} // namespace mlpack

#include "perceptron_impl.hpp"

#endif
