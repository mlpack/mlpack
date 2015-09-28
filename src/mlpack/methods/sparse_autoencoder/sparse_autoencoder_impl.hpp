/**
 * @file sparse_autoencoder_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of sparse autoencoders.
 */
#ifndef __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_IMPL_HPP
#define __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_autoencoder.hpp"

namespace mlpack {
namespace nn {

template<typename HiddenLayer, typename OutputLayer,
         template<typename> class OptimizerType>
SparseAutoencoder<HiddenLayer, OutputLayer, OptimizerType>::
SparseAutoencoder(const std::string &fileName,
                  const std::string& name)
{
  mlpack::data::Load(fileName, name,
                     *this, true);
}

template<typename HiddenLayer, typename OutputLayer,
         template<typename> class OptimizerType>
SparseAutoencoder<HiddenLayer, OutputLayer, OptimizerType>::
SparseAutoencoder(size_t visibleSize, size_t hiddenSize) :
  parameters(hiddenSize * 2 + 1, visibleSize + 1),
  visibleSize{visibleSize},
  hiddenSize{hiddenSize},
  lambda{0.0001},
  beta{3},
  rho{0.01}
{
}

template<typename HiddenLayer, typename OutputLayer,
         template<typename> class OptimizerType>
template<typename SparseAutoEncoderFunc>
SparseAutoencoder<HiddenLayer, OutputLayer, OptimizerType>::
SparseAutoencoder(OptimizerType<SparseAutoEncoderFunc> &optimizer) :
  parameters(optimizer.Function().GetInitialPoint()),
  visibleSize{optimizer.Function().VisibleSize()},
  hiddenSize{optimizer.Function().HiddenSize()},
  lambda{optimizer.Function().Lambda()},
  beta{optimizer.Function().Beta()},
  rho{optimizer.Function().Rho()}
{
  Train(optimizer);
}

template<typename HiddenLayer, typename OutputLayer,
         template<typename> class OptimizerType>
SparseAutoencoder<HiddenLayer, OutputLayer, OptimizerType>::
SparseAutoencoder(const arma::mat& data,
                  size_t visibleSize,
                  size_t hiddenSize,
                  double lambda,
                  double beta,
                  double rho) :
  visibleSize{visibleSize},
  hiddenSize{hiddenSize},
  lambda{lambda},
  beta{beta},
  rho{rho}
{
  SAEF encoderFunction(data, visibleSize, hiddenSize,
                       lambda, beta, rho);
  OptimizerType<SAEF> optimizer(encoderFunction);

  parameters = encoderFunction.GetInitialPoint();
  Train(optimizer);
}

template<typename HiddenLayer, typename OutputLayer,
         template<typename> class OptimizerType>
double SparseAutoencoder<HiddenLayer, OutputLayer, OptimizerType>::
Train(arma::mat const &data, size_t hiddenSize)
{
  SAEF encoderFunction(data, data.n_rows, hiddenSize,
                       lambda, beta, rho);
  OptimizerType<SAEF> optimizer(encoderFunction);
  Train(optimizer);
}

template<typename HiddenLayer, typename OutputLayer,
         template<typename> class OptimizerType>
template<typename SparseAutoEncoderFunc>
double SparseAutoencoder<HiddenLayer, OutputLayer, OptimizerType>::
Train(OptimizerType<SparseAutoEncoderFunc>& optimizer)
{
  Timer::Start("sparse_autoencoder_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("sparse_autoencoder_optimization");

  Log::Info << "SparseAutoencoder::SparseAutoencoder(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

template<typename HiddenLayer, typename OutputLayer,
         template<typename> class OptimizerType>
void SparseAutoencoder<HiddenLayer, OutputLayer, OptimizerType>::
GetNewFeatures(arma::mat& data,
               arma::mat& features)
{
  const size_t l1 = hiddenSize;
  const size_t l2 = visibleSize;

  Sigmoid(parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
          arma::repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols),
          features);
}

}; // namespace nn
}; // namespace mlpack

#endif
