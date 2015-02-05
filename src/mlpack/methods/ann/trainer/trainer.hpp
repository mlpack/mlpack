/**
 * @file trainer.hpp
 * @author Marcus Edel
 * @author Shangtong Zhang
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
  typename VecType = arma::colvec,
  typename InputType = arma::mat
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
     * @param maxEpochs The number of maximal trained iterations (0 means no
     * limit).
     * @param batchSize The batch size used to train the network.
     * @param tolerance Train the network until it converges against
     * the specified threshold.
     * @param shuffle If true, the order of the training set is shuffled;
     * otherwise, each data is visited in linear order.
     */
    Trainer(NetworkType& net,
            const size_t maxEpochs = 0,
            const size_t batchSize = 1,
            const double tolerance = 0.0001,
            const bool shuffle = true) :
        net(net),
        maxEpochs(maxEpochs),
        batchSize(batchSize),
        tolerance(tolerance),
        shuffle(shuffle)
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
    template <size_t dim = 1>
    typename std::enable_if<dim == 1, void>::type
    Train(InputType& trainingData,
               MatType& trainingLabels,
               InputType& validationData,
               MatType& validationLabels)
    {
      // This generates [0 1 2 3 ... (trainingData.n_cols - 1)]. The sequence
      // will be used to iterate through the training data.
      index = arma::linspace<arma::Col<size_t> >(0, trainingData.n_cols - 1,
                                                   trainingData.n_cols);

      epoch = 0;

      while(true)
      {

        // Randomly shuffle the index sequence if not in batch mode.
        if (shuffle)
          index = arma::shuffle(index);

        Train<dim>(trainingData, trainingLabels);
        Evaluate<dim>(validationData, validationLabels);

        if (validationError <= tolerance)
          break;

        if (maxEpochs > 0 && ++epoch > maxEpochs)
          break;
      }
    }
  
    // For 2-dim training data.
    template <size_t dim = 1>
    typename std::enable_if<dim == 2, void>::type
    Train(InputType& trainingData,
               MatType& trainingLabels,
               InputType& validationData,
               MatType& validationLabels)
    {
      // This generates [0 1 2 3 ... (trainingData.n_cols - 1)]. The sequence
      // will be used to iterate through the training data.
      index = arma::linspace<arma::Col<size_t> >(0, trainingData.n_slices - 1,
                                                 trainingData.n_slices);
    
      epoch = 0;
    
      while(true)
      {
      
        // Randomly shuffle the index sequence if not in batch mode.
        if (shuffle)
          index = arma::shuffle(index);
        
        Train<dim>(trainingData, trainingLabels);
        Evaluate<dim>(validationData, validationLabels);
        
        if (validationError <= tolerance)
          break;
        
        if (maxEpochs > 0 && ++epoch > maxEpochs)
          break;
      }
    }

    //! Get the training error.
    double TrainingError() const { return trainingError; }

    //! Get the validation error.
    double ValidationError() const { return validationError; }

    //! Get whether or not the individual inputs are shuffled.
    bool Shuffle() const { return shuffle; }
    //! Modify whether or not the individual inputs are shuffled.
    bool& Shuffle() { return shuffle; }

    //! Get the batch size.
    size_t StepSize() const { return batchSize; }
    //! Modify the batch size.
    size_t& StepSize() { return batchSize; }

    //! Get the maximum number of iterations (0 indicates no limit).
    size_t MaxEpochs() const { return maxEpochs; }
    //! Modify the maximum number of iterations (0 indicates no limit).
    size_t& MaxEpochs() { return maxEpochs; }

    //! Get the tolerance for termination.
    double Tolerance() const { return tolerance; }
    //! Modify the tolerance for termination.
    double& Tolerance() { return tolerance; }

  private:
  
    /**
     * Train the network on the given dataset.
     *
     * @param data Data used to train the network.
     * @param target Labels used to train the network.
     */
    template <size_t dim = 1>
    typename std::enable_if<dim == 1, void>::type
    Train(InputType& data, MatType& target)
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
  
    //For 2-dim training data.
    template <size_t dim = 1>
    typename std::enable_if<dim == 2, void>::type
    Train(InputType& data, MatType& target) {
      // Reset the training error.
      trainingError = 0;
      
      for (size_t i = 0; i < data.n_slices; i++)
      {
        net.FeedForward(data.slice(index(i)),
                        target.unsafe_col(index(i)), error);
        trainingError += net.Error();
        
        net.FeedBackward(error);
        
        if (((i + 1) % batchSize) == 0)
          net.ApplyGradients();
      }
      
      if ((data.n_slices % batchSize) != 0)
        net.ApplyGradients();
      
      trainingError /= data.n_slices;
    }

    /**
     * Evaluate the network on the given dataset.
     *
     * @param data Data used to train the network.
     * @param target Labels used to train the network.
     */
    template <size_t dim = 1>
    typename std::enable_if<dim == 1, void>::type
    Evaluate(InputType& data, MatType& target)
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
  
    //For 2-dim training data.
    template <size_t dim = 1>
    typename std::enable_if<dim == 2, void>::type
    Evaluate(InputType& data, MatType& target) {
      validationError = 0;
      
      for (size_t i = 0; i < data.n_slices; i++)
      {
        net.FeedForward(data.slice(i), target.unsafe_col(i), error);
        validationError += net.Error();
      }
      
      validationError /= data.n_slices;
    }

    //! The network which should be trained and evaluated.
    NetworkType& net;

    //! The current network error of a single input.
    typename std::conditional<NetworkTraits<NetworkType>::IsFNN ||
        NetworkTraits<NetworkType>::IsCNN,
        VecType, MatType>::type error;

    //! The current epoch if maxEpochs is set.
    size_t epoch;

    //! The maximal epochs that should be used.
    size_t maxEpochs;

    //! The size until a update is performed.
    size_t batchSize;

    //! The shuffel sequence index used to train the network.
    arma::Col<size_t> index;

    //! The overall traing error.
    double trainingError;

    //! The overall validation error.
    double validationError;

    //! The tolerance for termination.
    double tolerance;

    //! Controls whether or not the individual inputs are shuffled when
    //! iterating.
    bool shuffle;
}; // class Trainer

}; // namespace ann
}; // namespace mlpack

#endif
