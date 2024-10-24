/**
 * @file methods/perceptron/perceptron.hpp
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

#include "initialization_methods/initialization_methods.hpp"
#include "learning_policies/learning_policies.hpp"

namespace mlpack {

/**
 * This class implements a simple perceptron (i.e., a single layer neural
 * network).  It converges if the supplied training dataset is linearly
 * separable.
 *
 * @tparam LearnPolicy Options of SimpleWeightUpdate and GradientDescent.
 * @tparam WeightInitializationPolicy Option of ZeroInitialization and
 *      RandomPerceptronInitialization.
 */
template<typename LearnPolicy = SimpleWeightUpdate,
         typename WeightInitializationPolicy = ZeroInitialization,
         typename MatType = arma::mat>
class Perceptron
{
 public:
  //! The element type used in the Perceptron.
  using ElemType = typename MatType::elem_type;

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
   * Constructor: construct the perceptron by building the weights matrix, which
   * is later used in classification.  The number of classes should be specified
   * separately, and the labels vector should contain values in the range [0,
   * numClasses - 1].  The data::NormalizeLabels() function can be used if the
   * labels vector does not contain values in the required range.
   *
   * This constructor supports weights for each data point.
   *
   * @param data Input, training data.
   * @param labels Labels of dataset.
   * @param numClasses Number of classes in the dataset.
   * @param instanceWeights Weight vector to use for each training point while
   *     training.
   * @param maxIterations Maximum number of iterations for the perceptron
   *     learning algorithm.
   */
  template<typename WeightsType>
  Perceptron(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const WeightsType& instanceWeights,
             const size_t maxIterations = 1000,
             const std::enable_if_t<
                 arma::is_arma_type<WeightsType>::value>* = 0);

  /**
   * Alternate constructor which copies parameters from an already initiated
   * perceptron.
   *
   * @param other The other initiated Perceptron object from which we copy the
   *       values from.
   * @param data The data on which to train this Perceptron object on.
   * @param labels The labels of data.
   * @param numClasses Number of classes in the data.
   * @param instanceWeights Weight vector to use while training. For boosting
   *      purposes.
   */
  template<typename WeightsType>
  [[deprecated("Use other constructors")]]
  Perceptron(const Perceptron& other,
             const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const WeightsType& instanceWeights,
             const std::enable_if_t<
                 arma::is_arma_type<WeightsType>::value>* = 0);

  /**
   * Train the perceptron on the given data for up to the given maximum number
   * of iterations.  A single iteration corresponds to a single pass through the
   * data, so if you want to pass through the dataset only once, set
   * `maxIterations` to 1.
   *
   * After calling this overload, `MaxIterations()` will return whatever
   * `maxIterations` was given to this function.
   *
   * This training does not reset the model weights, so you can call Train() on
   * multiple datasets sequentially.
   *
   * @param data Dataset on which training should be performed.
   * @param labels Labels of the dataset.
   * @param numClasses Number of classes in the data.
   * @param maxIterations Maximum number of iterations for training.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const std::optional<size_t> maxIterations = std::nullopt);

  /**
   * Train the perceptron on the given data for up to the given maximum number
   * of iterations.  A single iteration corresponds to a single pass through the
   * data, so if you want to pass through the dataset only once, set
   * `maxIterations` to 1.
   *
   * After calling this overload, `MaxIterations()` will return whatever
   * `maxIterations` was given to this function.
   *
   * This training does not reset the model weights, so you can call Train() on
   * multiple datasets sequentially.
   *
   * @param data Dataset on which training should be performed.
   * @param labels Labels of the dataset.
   * @param numClasses Number of classes in the data.
   * @param instanceWeights Cost matrix. Stores the cost of mispredicting
   *      instances.  This is useful for boosting.
   * @param maxIterations Maximum number of iterations for training.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const arma::rowvec& instanceWeights,
             const std::optional<size_t> maxIterations = std::nullopt);

  /**
   * After training, use the weights matrix to classify `point`, and return the
   * predicted class.
   *
   * @param point Test point to classify.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Classification function. After training, use the weights matrix to
   * classify test, and put the predicted classes in predictedLabels.
   *
   * @param test Testing data or data to classify.
   * @param predictedLabels Vector to store the predicted classes after
   *     classifying test.
   */
  void Classify(const MatType& test, arma::Row<size_t>& predictedLabels) const;

  /**
   * Reset the model, so that the next call to `Train()` will not be
   * incremental.
   */
  void Reset();

  /**
   * Serialize the perceptron.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the number of classes this perceptron has been trained for.
  size_t NumClasses() const { return weights.n_cols; }

  //! Get the weight matrix.
  const arma::Mat<ElemType>& Weights() const { return weights; }
  //! Modify the weight matrix.  You had better know what you are doing!
  arma::Mat<ElemType>& Weights() { return weights; }

  //! Get the biases.
  const arma::Col<ElemType>& Biases() const { return biases; }
  //! Modify the biases.  You had better know what you are doing!
  arma::Col<ElemType>& Biases() { return biases; }

 private:
  /**
   * Internal training function; this assumes that maxIterations has been set.
   *
   * If `HasWeights` is `false`, then `instanceWeights` is ignored (and may be
   * left empty).
   */
  template<bool HasWeights, typename WeightsType>
  void TrainInternal(const MatType& data,
                     const arma::Row<size_t>& labels,
                     const size_t numClasses,
                     const WeightsType& instanceWeights = WeightsType());

  //! The maximum number of iterations during training.
  size_t maxIterations;

  /**
   * Stores the weights for each of the input class labels.  Each column
   * corresponds to the weights for one class label, and each row corresponds to
   * the weights for one dimension of the input data.  The biases are held in a
   * separate vector.
   */
  arma::Mat<ElemType> weights;

  //! The biases for each class.
  arma::Col<ElemType> biases;
};

} // namespace mlpack

#include "perceptron_impl.hpp"

#endif
