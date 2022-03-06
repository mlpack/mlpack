/**
 * @file methods/rvm_regression_model_impl.hpp
 * @author Clement Mercier
 *
 * A serializable RVM model used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RVM_REGRESSION_MODEL_IMPL_HPP
#define MLPACK_METHODS_RVM_REGRESSION_MODEL_IMPL_HPP

#include "rvm_regression_model.hpp"

using namespace mlpack;

template<typename KernelType>
Wrapper<KernelType>::Wrapper(const KernelType& kernel,
	const bool centerData,
	const bool scaleData,
	const bool ard)
{
  rvm = RVMRegression<KernelType>(kernel,
				  centerData,
				  scaleData,
				  ard);
}

template<typename KernelType>
void Wrapper<KernelType>::Train(const mat& matX, const rowvec& responses)
{
  rvm.Train(matX, responses);
}

template<typename KernelType>
void Wrapper<KernelType>::Predict(const mat& matX, rowvec& predictions)
{
  rvm.Predict(matX, predictions);
}

template<typename KernelType>
void Wrapper<KernelType>::Predict(const mat& matX, rowvec& predictions,
	     rowvec& std)
{
  rvm.Predict(matX, predictions, std);
}

template<typename KernelType>
const RVMRegression<KernelType>* Wrapper<KernelType>::GetRVMptr() const
{
  return &rvm;
}

template<typename Archive>
void SerializeHelperRVM(Archive& ar, WrapperBase* rvmWrapper, const Kernel kernel_)
{
  switch(kernel_)
  {
    case LINEAR:
    {
      Wrapper<LinearKernel>& rvm_ =
	dynamic_cast<Wrapper<LinearKernel>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
    case GAUSSIAN:
    {
      Wrapper<GaussianKernel>& rvm_ =
	dynamic_cast<Wrapper<GaussianKernel>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
    case LAPLACIAN:
    {
      Wrapper<LaplacianKernel>& rvm_ =
	dynamic_cast<Wrapper<LaplacianKernel>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
    case EPANECHNIKOV:
    {
      Wrapper<EpanechnikovKernel>& rvm_ =
	dynamic_cast<Wrapper<EpanechnikovKernel>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
    case SPHERICAL:
    {
      Wrapper<SphericalKernel>& rvm_ =
	dynamic_cast<Wrapper<SphericalKernel>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
    case POLYNOMIAL:
    {
      Wrapper<PolynomialKernel>& rvm_ =
	dynamic_cast<Wrapper<PolynomialKernel>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
    case COSINE:
    {
      Wrapper<CosineDistance>& rvm_ =
	dynamic_cast<Wrapper<CosineDistance>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
    case ARD:
    {
      Wrapper<LinearKernel>& rvm_ =
	dynamic_cast<Wrapper<LinearKernel>&>(*rvmWrapper);
      ar(CEREAL_NVP(rvm_));
      break;
    }
  }
}

template<typename Archive>
void RVMRegressionModel::serialize(Archive& ar, const uint32_t /* version */)
{
  SerializeHelperRVM(ar, rvmWrapper, kernel_);
}

#endif

