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
   * Constructor - constructs the perceptron by building the weights
   * matrix, which is later used in Classification.  It adds a bias input vector
   * of 1 to the input data to take care of the bias weights.
   *
   * @param data Input, training data.
   * @param labels Labels of dataset.
   * @param iterations Maximum number of iterations for the perceptron learning
   *     algorithm.
   */
  Perceptron(const MatType& data,
             const arma::Row<size_t>& labels,
             const int maxIterations);

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
             const arma::rowvec& D,
             const arma::Row<size_t>& labels);

  /**
   * Serialize the perceptron.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

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

  /**
   * Training Function. It trains on trainData using the cost matrix D
   *
   * @param D Cost matrix. Stores the cost of mispredicting instances
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const arma::rowvec& D = arma::rowvec());
};

} // namespace perceptron
} // namespace mlpack

#include "perceptron_impl.hpp"

#endif
