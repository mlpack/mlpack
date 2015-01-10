/**
 * @file trainer.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of a trainer that trains the parameters of a
 * neural network according to a supervised dataset.
 */
#ifndef __MLPACK_METHODS_ANN_TRAINER_TRAINER_HPP
#define __MLPACK_METHODS_ANN_TRAINER_TRAINER_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/network_traits.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/neuron_layer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Trainer that trains the parameters of a neural network according to a
 * supervised dataset.
 *
 * @tparam NetworkType The type of network which should be trained and
 * evaluated.
 * @tparam MaType Type of the error type (arma::mat or arma::sp_mat).
 * @tparam VecType Type of error type (arma::colvec, arma::mat or arma::sp_mat).
 */
template<
  typename NetworkType,
  typename MatType = arma::mat,
  typename VecType = arma::colvec
>
class Trainer
{
  public:
    /**
     * Construct the Trainer object, which will be used to train a neural
     * network according to a supervised dataset by backpropagating the errors.
     *
     * If batchSize is greater 1 the trainer will take a mean gradient step over
     * this many samples and will update the parameters only at the end of
     * each epoch (Default 1).
     *
     * @param net The network that should be trained.
     * @param maxEpochs The number of maximal trained iterations.
     * @param batchSize The batch size used to train the network.
     * @param convergenceThreshold Train the network until it converges against
     * the specified threshold.
     */
    Trainer(NetworkType& net,
            const size_t maxEpochs = 0,
            const size_t batchSize = 1,
            const double convergenceThreshold  = 0.0001) :
        net(net),
        maxEpochs(maxEpochs),
        batchSize(batchSize),
        convergenceThreshold(convergenceThreshold)
    {
      // Nothing to do here.
    }

    /**
     * Train the network on the given datasets until the network converges. If
     * maxEpochs is greater than zero that many epochs are maximal trained.
     *
     * @param trainingData Data used to train the network.
     * @param trainingLabels Labels used to train the network.
     * @param validationData Data used to evaluate the network.
     * @tparam validationLabels Labels used to evaluate the network.
     */
    void Train(MatType& trainingData,
               MatType& trainingLabels,
               MatType& validationData,
               MatType& validationLabels)
    {
      // This generates [0 1 2 3 ... (trainingData.n_cols - 1)]. The sequence
      // will be used to iterate through the training data.
      index = arma::linspace<arma::Col<size_t> >(0, trainingData.n_cols - 1,
          trainingData.n_cols);
      epoch = 0;

      while(true)
      {
        // Randomly shuffle the index sequence.
        index = arma::shuffle(index);

        Train(trainingData, trainingLabels);
        Evaluate(validationData, validationLabels);

        if (validationError <= convergenceThreshold)
          break;

        if (maxEpochs > 0 && ++epoch > maxEpochs)
          break;
      }
    }

    //! Get the training error.
    double TrainingError() const { return trainingError; }

    //! Get the validation error.
    double ValidationError() const { return validationError; }

  private:
    /**
     * Train the network on the given dataset.
     *
     * @param data Data used to train the network.
     * @param target Labels used to train the network.
     */
    void Train(MatType& data, MatType& target)
    {
      // Reset the training error.
      trainingError = 0;

      for (size_t i = 0; i < data.n_cols; i++)
      {
        net.FeedForward(data.unsafe_col(index(i)),
            target.unsafe_col(index(i)), error);
        trainingError += net.Error();

        net.FeedBackward(error);

        if (((i + 1) % batchSize) == 0)
          net.ApplyGradients();
      }

      if ((data.n_cols % batchSize) != 0)
        net.ApplyGradients();

      trainingError /= data.n_cols;
    }

    /**
     * Evaluate the network on the given dataset.
     *
     * @param data Data used to train the network.
     * @param target Labels used to train the network.
     */
    void Evaluate(MatType& data, MatType& target)
    {
      // Reset the validation error.
      validationError = 0;

      for (size_t i = 0; i < data.n_cols; i++)
      {
        net.FeedForward(data.unsafe_col(i), target.unsafe_col(i), error);
        validationError += net.Error();
      }

      validationError /= data.n_cols;
    }

    //! The network which should be trained and evaluated.
    NetworkType& net;

    //! The current network error of a single input.
    typename std::conditional<NetworkTraits<NetworkType>::IsFNN,
        VecType, MatType>::type error;

    //! The current epoch if maxEpochs is set.
    size_t epoch;

    //! The maximal epochs that should be used.
    size_t maxEpochs;

    //! The size until a update is performed.
    size_t batchSize;

    //! Index sequence used to train the network.
    arma::Col<size_t> index;

    //! The overall traing error.
    double trainingError;

    //! The overall validation error.
    double validationError;

    //! The threshold used as convergence.
    double convergenceThreshold;
}; // class Trainer

}; // namespace ann
}; // namespace mlpack

#endif
