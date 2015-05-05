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
    template<typename InputType, typename OutputType>
    void Train(InputType& trainingData,
               OutputType& trainingLabels,
               InputType& validationData,
               OutputType& validationLabels)
    {
      // This generates [0 1 2 3 ... (ElementCount(trainingData) - 1)]. The
      // sequence will be used to iterate through the training data.
      index = arma::linspace<arma::Col<size_t> >(0,
          ElementCount(trainingData) - 1, ElementCount(trainingData));
      epoch = 0;

      while(true)
      {
        if (shuffle)
          index = arma::shuffle(index);

        Train(trainingData, trainingLabels);
        Evaluate(validationData, validationLabels);

        if (validationError <= tolerance)
          break;

        if (maxEpochs > 0 && ++epoch >= maxEpochs)
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
    template<typename InputType, typename OutputType>
    void Train(InputType& data, OutputType& target)
    {
      // Reset the training error.
      trainingError = 0;

      for (size_t i = 0; i < index.n_elem; i++)
      {
        net.FeedForward(Element(data, index(i)),
            Element(target, index(i)), error);

        trainingError += net.Error();
        net.FeedBackward(error);

        if (((i + 1) % batchSize) == 0)
          net.ApplyGradients();
      }

      if ((index.n_elem % batchSize) != 0)
        net.ApplyGradients();

      trainingError /= index.n_elem;
    }

    /**
     * Evaluate the network on the given dataset.
     *
     * @param data Data used to train the network.
     * @param target Labels used to train the network.
     */
    template<typename InputType, typename OutputType>
    void Evaluate(InputType& data, OutputType& target)
    {
      // Reset the validation error.
      validationError = 0;

      for (size_t i = 0; i < ElementCount(data); i++)
      {
         validationError += net.Evaluate(Element(data, i),
            Element(target, i), error);
      }

      validationError /= ElementCount(data);
    }

    /*
     * Create a Col object which uses memory from an existing matrix object.
     * (This approach is currently not alias safe)
     *
     * @param data The reference data.
     * @param sliceNum Provide a Col object of the specified index.
     */
    template<typename eT>
    arma::Col<eT> Element(arma::Mat<eT>& input, const size_t colNum)
    {
      return arma::Col<eT>(input.colptr(colNum), input.n_rows, false, true);
    }

    /*
     * Provide the reference to the matrix representing a single slice.
     *
     * @param data The reference data.
     * @param sliceNum Provide a single slice of the specified index.
     */
    template<typename eT>
    const arma::Mat<eT>& Element(arma::Cube<eT>& input, const size_t sliceNum)
    {
      return *(input.mat_ptrs[sliceNum]);
    }

    /*
     * Get the number of elements.
     *
     * @param data The reference data.
     */
    template<typename eT>
    size_t ElementCount(const arma::Mat<eT>& data) const
    {
      return data.n_cols;
    }

    /*
     * Get the number of elements.
     *
     * @param data The reference data.
     */
    template<typename eT>
    size_t ElementCount(const arma::Cube<eT>& data) const
    {
      return data.n_slices;
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
