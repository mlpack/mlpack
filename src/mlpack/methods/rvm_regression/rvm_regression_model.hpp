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

#include <mlpack/methods/rvm_regression/rvm_regression.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include "boost/variant.hpp"

using namespace mlpack::kernel;
using namespace mlpack::regression;

/**
 * Serializabale RVM class.
 */
class RVMRegressionModel
{
  private:
  boost::variant<RVMRegression<LinearKernel>*,
		 RVMRegression<GaussianKernel>*> rVariant;

 public:
 RVMRegressionModel(const std::string kernelType,
		    const bool centerData,
		    const bool scaleData,
		    const double bandwidth = 1.0);

  //! Create an empty CF model.
  RVMRegressionModel() { }

 // Clean up memory.
  ~RVMRegressionModel();
  
  void Train(const arma::mat& matX, const arma::rowvec& responses);
  
  void Predict(const arma::mat& matX, arma::rowvec& predictions);
  
  void Predict(const arma::mat& matX, arma::rowvec& predictions,
      arma::rowvec& std);

  //! Get the pointer to RVMRegression<> object.
  template <typename KernelType>
  const RVMRegression<KernelType>* RVMPtr() const;

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
};

/**
 * TrainVisitor train the RVMRegression model whatever its kernel type.
 */
class TrainVisitor : public boost::static_visitor<>
{
  private:
  const arma::mat& matX;
  const arma::rowvec& responses;
  
  public:
  // Train the RVMRegression model.
  TrainVisitor(const arma::mat& matX, const arma::rowvec& responses);

  // Generic visitor for training.
  template <typename T>
  void operator()(T* t) const;
};

/**
 * PredictVisitor makes predictions on a given dataset.
 */
class PredictVisitor : public boost::static_visitor<>
{
  private:
    const arma::mat& matX;
    arma::rowvec& predictions;
  
  public:
    PredictVisitor(const arma::mat& matX,
		   arma::rowvec& predictions);

  template <typename T>
  void operator()(T* t) const;
};

/**
 * PredictStdVisitor makes predictions on a given dataset with uncertainies.
 */
class PredictStdVisitor : public boost::static_visitor<>
{
private:
  const arma::mat& matX;
  arma::rowvec& predictions;
  arma::rowvec& std;
  
public:
  PredictStdVisitor(const arma::mat& matX,
		    arma::rowvec& predictions,
		    arma::rowvec& std);

  template <typename T>
  void operator()(T* t) const;
};

/**
 * GetValueVisitor returns the pointer which points to the CFType object.
 */
class GetValueVisitor : public boost::static_visitor<void*>
{
 public:
  //! Return stored pointer as void* type.
  template <typename T> void* operator()(T *t) const;
};

struct DeleteVisitor : public boost::static_visitor<>
{
  template <typename T>
  void operator()(T* t) const
  {
    if (!t)
      delete t;
  }
};

#include "rvm_regression_model_impl.hpp"

#endif
