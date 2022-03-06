/**
 * @file methods/rvm_regression_model.hpp
 * @author Clement Mercier
 *
 * A serializable RVM model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RVM_REGRESSION_MODEL_HPP
#define MLPACK_METHODS_RVM_REGRESSION_MODEL_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/rvm_regression/rvm_regression.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
#include <mlpack/core/kernels/cosine_distance.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/spherical_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>



using namespace mlpack::kernel;
using namespace mlpack::regression;
using namespace arma;

namespace mlpack {

enum Kernel
{
 LINEAR,
 GAUSSIAN,
 LAPLACIAN,
 EPANECHNIKOV,
 SPHERICAL,
 POLYNOMIAL,
 COSINE,
 ARD
};
  
class WrapperBase
{
  public:
  //! Create the object. The base class has nothing to hold.
  WrapperBase() {};

  //! Delete the object.
  virtual ~WrapperBase() {};

  //! Make a training.
  virtual void Train(const mat& matX, const rowvec& responses) = 0;

  //! Compute predictions from matX.
  virtual void Predict(const mat& matX, rowvec& predictions) = 0;

  //! Compute predictions and uncertainties from matX.
  virtual void Predict(const mat& matX, rowvec& predictions, rowvec& std) = 0;

  //! Get solution vector.
  virtual const colvec& Omega() const = 0;
};

template<typename KernelType>
class Wrapper : public WrapperBase
{
  protected:
  typedef RVMRegression<KernelType> RVMRType;
  RVMRType rvm;

  public:
  Wrapper(const KernelType& kernel,
	  const bool centerData,
	  const bool scaleData,
	  const bool ard);

  //! Delete the Wrapper object.
  virtual ~Wrapper() {};
  
  //! Make a training.
  virtual void Train(const mat& matX, const rowvec& responses);

  //! Compute predictions from matX.
  virtual void Predict(const mat& matX, rowvec& predictions);

  //! Compute predictions and uncertainties from matX.
  virtual void Predict(const mat& matX, rowvec& predictions, rowvec& std);

  //! Get internal pointer to RVMRegression<> object.
  const RVMRegression<KernelType>* GetRVMptr() const;
  
  //! Get solution vector.
  virtual const colvec& Omega() const {return rvm.Omega();}

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(rvm));
  }

};

class RVMRegressionModel
{
  public:

  //! Create an empty model.
  RVMRegressionModel();
  
  //! Create the RVMRegressionModel.
  RVMRegressionModel(const std::string kernelType,
		     const bool centerData,
		     const bool scaleData,
		     const double bandwidth = 1.0,
		     const double offset = 0.0,
		     const double kernel_scale = 1.0,
		     const double degree = 2.0);

  //! Make a training.
  void Train(const mat& matX, const rowvec& responses);

  //! Compute predictions from matX.
  void Predict(const mat& matX, rowvec& predictions);

  //! Compute predictions and uncertainties from matX.
  void Predict(const mat& matX, rowvec& predictions, rowvec& std);

  //! Get solution vector.
  inline const colvec& Omega() const { return rvmWrapper->Omega(); }

  //! Get a pointer to the internal RVMRegression object.
  template<typename KernelType>
  const RVMRegression<KernelType>* RVMPtr() const
  {
    Wrapper<KernelType>* rvm_ = dynamic_cast<Wrapper<KernelType>*>(rvmWrapper);
    return rvm_->GetRVMptr();
  }

  //! Destructor.
  ~RVMRegressionModel();

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
  
  private:
  Kernel kernel_;
  WrapperBase* rvmWrapper;

};

} // namespace mlpack

#include "rvm_regression_model_impl.hpp"

#endif
