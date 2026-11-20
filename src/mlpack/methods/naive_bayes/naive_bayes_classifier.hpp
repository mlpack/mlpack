/**
 * @file methods/naive_bayes/naive_bayes_classifier.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Shihao Jing (shihao.jing810@gmail.com)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  It is assumed that the features have been sampled from a
 * Gaussian PDF.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP
#define MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * The simple Naive Bayes classifier.  This class trains on the data by
 * calculating the sample mean and variance of the features with respect to each
 * of the labels, and also the class probabilities.  The class labels are
 * assumed to be positive integers (starting with 0), and are expected to be the
 * last row of the data input to the constructor.
 *
 * Mathematically, it computes P(X_i = x_i | Y = y_j) for each feature X_i for
 * each of the labels y_j.  Along with this, it also computes the class
 * probabilities P(Y = y_j).
 *
 * For classifying a data point (x_1, x_2, ..., x_n), it computes the following:
 * arg max_y(P(Y = y)*P(X_1 = x_1 | Y = y) * ... * P(X_n = x_n | Y = y))
 *
 * Example use:
 *
 * @code
 * extern arma::mat training_data, testing_data;
 * NaiveBayesClassifier<> nbc(training_data, 5);
 * arma::vec results;
 *
 * nbc.Classify(testing_data, results);
 * @endcode
 *
 * The ModelMatType template parameter specifies the internal matrix type that
 * NaiveBayesClassifier will use to hold the means, variances, and weights that
 * make up the Naive Bayes model.  This can be arma::mat, arma::fmat, or any
 * other Armadillo (or Armadillo-compatible) object.  Because ModelMatType may
 * be different than the type of the data the model is trained on, now training
 * is possible with subviews, sparse matrices, or anything else, while still
 * storing the model as a ModelMatType internally.
 *
 * @tparam ModelMatType Internal matrix type to use to store the model.
 */
template<typename ModelMatType = arma::mat>
class NaiveBayesClassifier
{
 public:
  // Convenience typedef.
  using ElemType = typename ModelMatType::elem_type;

  /**
   * Initializes the classifier as per the input and then trains it by
   * calculating the sample mean and variances.
   *
   * Example use:
   * @code
   * extern arma::mat training_data, testing_data;
   * extern arma::Row<size_t> labels;
   * NaiveBayesClassifier nbc(training_data, labels, 5);
   * @endcode
   *
   * @param data Training data points.
   * @param labels Labels corresponding to training data points.
   * @param numClasses Number of classes in this classifier.
   * @param incrementalVariance If true, an incremental algorithm is used to
   *     calculate the variance; this can prevent loss of precision in some
   *     cases, but will be somewhat slower to calculate.
   * @param epsilon Small initialization value for variances to prevent log of
   *     zero.
   */
  template<typename MatType>
  NaiveBayesClassifier(const MatType& data,
                       const arma::Row<size_t>& labels,
                       const size_t numClasses,
                       const bool incrementalVariance = false,
                       const double epsilon = 1e-10);

  /**
   * Initialize the Naive Bayes classifier without performing training.  All of
   * the parameters of the model will be initialized to zero.  Be sure to use
   * Train() before calling Classify(), otherwise the results may be
   * meaningless.
   */
  NaiveBayesClassifier(const size_t dimensionality = 0,
                       const size_t numClasses = 0,
                       const double epsilon = 1e-10);

  /**
   * Train the Naive Bayes classifier on the given dataset.  If the incremental
   * algorithm is used, the current model is used as a starting point (this is
   * the default).  If the incremental algorithm is not used, then the current
   * model is ignored and the new model will be trained only on the given data.
   * Note that even if the incremental algorithm is not used, the data must have
   * the same dimensionality and number of classes that the model was
   * initialized with.  If you want to change the dimensionality or number of
   * classes, either re-initialize or call Means(), Variances(), and
   * Probabilities() individually to set them to the right size.
   *
   * @param data The dataset to train on.
   * @param labels The labels for the dataset.
   * @param numClasses The number of classes in the dataset.
   * @param incremental Whether or not to use the incremental algorithm for
   *      training.
   */
  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const bool incremental = true);

  /**
   * Train the Naive Bayes classifier on the given dataset.  If the incremental
   * algorithm is used, the current model is used as a starting point (this is
   * the default).  If the incremental algorithm is not used, then the current
   * model is ignored and the new model will be trained only on the given data.
   * Note that even if the incremental algorithm is not used, the data must have
   * the same dimensionality and number of classes that the model was
   * initialized with.  If you want to change the dimensionality or number of
   * classes, either re-initialize or call Means(), Variances(), and
   * Probabilities() individually to set them to the right size.
   *
   * @param data The dataset to train on.
   * @param labels The labels for the dataset.
   * @param numClasses The number of classes in the dataset.
   * @param incremental Whether or not to use the incremental algorithm for
   *      training.
   * @param epsilon Small reinitialization value for variances to prevent log of
   *      zero (ignored if incremental is true).
   */
  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const bool incremental,
             const double epsilon);

  /**
   * Train the Naive Bayes classifier on the given point.  This will use the
   * incremental algorithm for updating the model parameters.  The data must be
   * the same dimensionality as the existing model parameters.
   *
   * @param point Data point to train on.
   * @param label Label of data point.
   */
  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  /**
   * Classify the given point, using the trained NaiveBayesClassifier model. The
   * predicted label is returned.
   *
   * @param point Point to classify.
   */
  template<typename VecType>
  size_t Classify(const VecType& point) const;

  /**
   * Classify the given point using the trained NaiveBayesClassifier model and
   * also return estimates of the probability for each class in the given
   * vector.
   *
   * @param point Point to classify.
   * @param prediction This will be set to the predicted class of the point.
   * @param probabilities This will be filled with class probabilities for the
   *      point.
   */
  template<typename VecType, typename ProbabilitiesVecType>
  void Classify(const VecType& point,
                size_t& prediction,
                ProbabilitiesVecType& probabilities) const;

  /**
   * Classify the given points using the trained NaiveBayesClassifier model.
   * The predicted labels for each point are stored in the given vector.
   *
   * @code
   * arma::mat test_data; // each column is a test point
   * arma::Row<size_t> results;
   * ...
   * nbc.Classify(test_data, results);
   * @endcode
   *
   * @param data List of data points.
   * @param predictions Vector that class predictions will be placed into.
   */
  template<typename MatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions) const;

  /**
   * Classify the given points using the trained NaiveBayesClassifier model and
   * also return estimates of the probabilities for each class in the given
   * matrix.  The predicted labels for each point are stored in the given
   * vector.
   *
   * @code
   * arma::mat test_data; // each column is a test point
   * arma::Row<size_t> results;
   * arma::mat resultsProbs;
   * ...
   * nbc.Classify(test_data, results, resultsProbs);
   * @endcode
   *
   * @param data Set of points to classify.
   * @param predictions This will be filled with predictions for each point.
   * @param probabilities This will be filled with class probabilities for each
   *      point. Each row represents a point.
   * @tparam MatType Type of data to be classified.
   * @tparam ProbabilitiesMatType Type to store output probabilities in.
   */
  template<typename MatType, typename ProbabilitiesMatType>
  void Classify(const MatType& data,
                arma::Row<size_t>& predictions,
                ProbabilitiesMatType& probabilities) const;

  /**
   * Reset the model to zeros, keeping the model's current dimensionality and
   * number of classes.
   */
  void Reset();

  /**
   * Reset the model to zeros, with a new dimensionality and number of classes.
   * The value epsilon specifies an initial very small value for the variances,
   * to prevent log(0) issues.
   */
  void Reset(const size_t dimensionality,
             const size_t numClasses,
             const double epsilon = 1e-10);

  //! Get the sample means for each class.
  const ModelMatType& Means() const { return means; }
  //! Modify the sample means for each class.
  ModelMatType& Means() { return means; }

  //! Get the sample variances for each class.
  const ModelMatType& Variances() const { return variances; }
  //! Modify the sample variances for each class.
  ModelMatType& Variances() { return variances; }

  //! Get the prior probabilities for each class.
  const ModelMatType& Probabilities() const { return probabilities; }
  //! Modify the prior probabilities for each class.
  ModelMatType& Probabilities() { return probabilities; }

  //! Get the number of points the model has been trained on so far.
  size_t TrainingPoints() const { return trainingPoints; }

  //! Serialize the classifier.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! Sample mean for each class.
  ModelMatType means;
  //! Sample variances for each class.
  ModelMatType variances;
  //! Class probabilities; this has the shape of a column vector.
  ModelMatType probabilities;
  //! Number of training points seen so far.
  size_t trainingPoints;
  //! Small value to prevent log of zero.
  double epsilon;

  /**
   * Compute the unnormalized posterior log probability of given points (log
   * likelihood). Results are returned as arma::mat, and each column represents
   * a point, each row represents log likelihood of a class.
   *
   * @param data Set of points to compute posterior log probability for.
   * @param logLikelihoods Matrix to store log likelihoods in.
   */
  template<typename MatType>
  void LogLikelihood(const MatType& data,
                     ModelMatType& logLikelihoods) const;
};

} // namespace mlpack

CEREAL_TEMPLATE_CLASS_VERSION((typename MatType),
    (mlpack::NaiveBayesClassifier<MatType>), (1));

// Include implementation.
#include "naive_bayes_classifier_impl.hpp"

#endif
