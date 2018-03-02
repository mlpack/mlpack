/**
 * @file variance_scaling_init.hpp
 * @author Shashank Shekhar
 *
 * Intialization rule for the neural networks.
 * VarianceScalingNormalInit is performed by assigning a gaussian matrix with
 * zero mean and variance given by (scaling factor / n) to the weight matrix.
 * VarianceScalingUniformInit is performed by assigning a random matrix between
 * [-limit, limit] where limit is given as sqrt(3*scaling facor/n).
 * n is decided by the mode of initializer and is described below.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_VARIANCE_SCALING_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_VARIANCE_SCALING_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

#include "gaussian_init.hpp"
#include "random_init.hpp"


namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

class FanIn
{};
class FanOut
{};
class FanAvg
{};


/**
 * This class is used to initialize weigth matrix from a gaussian distribution
 * with zero mean and variance decided by mode type. The variance is given by
 * (scaling factor/n) where n is as follows :
 * n = rows for FanIn mode (number of neurons feeding in)
 * n = cols for FanOut mode (number of neurons feeding out)
 * n = (rows + cols)/2 for FanAvg mode (average of neurons feeding in and out)
 */
template<typename ModeType = FanAvg>
class VarianceScalingNormalInit
{
 public:
  // Empty Constructor
  // Scaling factor is the amount by which variance will be scaled.
  VarianceScalingNormalInit(const size_t scalingFactor = 1):
  scalingFactor(scalingFactor)
  {}

  template<typename eT, typename Mode = ModeType>
  void Initialize(
      arma::Mat<eT>& W,
      const size_t rows,
      const size_t cols,
      typename std::enable_if_t<std::is_same<Mode, FanIn>::value>* = 0)
  {
    double var = (scalingFactor / (double) (rows));
    GaussianInitialization init(0, var);
    init.Initialize(W, rows, cols);
  }

  template<typename eT, typename Mode = ModeType>
  void Initialize(
      arma::Mat<eT>& W,
      const size_t rows,
      const size_t cols,
      typename std::enable_if_t<std::is_same<Mode, FanOut>::value>* = 0)
  {
    double var = (scalingFactor / (double) (cols));
    GaussianInitialization init(0, var);
    init.Initialize(W, rows, cols);
  }

  template<typename eT, typename Mode = ModeType>
  void Initialize(
      arma::Mat<eT>& W,
      const size_t rows,
      const size_t cols,
      typename std::enable_if_t<std::is_same<Mode, FanAvg>::value>* = 0)
  {
    double var = (2 * scalingFactor / ((double) (rows) + (double) (cols)));
    GaussianInitialization init(0, var);
    init.Initialize(W, rows, cols);
  }


  /**
   * Initialize the elements of the specified weight 3rd order tensor.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slice Numbers of slices.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::Cube<eT>(rows, cols, slices);

    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  const size_t scalingFactor;
}; // class VarianceScalingNormalInit


/**
 * This class is used to initialize weight matrix from a random distribution
 * with bound = [-limit,limit] where limit is decided by mode type. The limit
 * is given by sqrt(3*scaling factor/n) where n is as follows :
 * n = rows for FanIn mode (number of neurons feeding in)
 * n = cols for FanOut mode (number of neurons feeding out)
 * n = (rows + cols)/2 for FanAvg mode (average of neurons feeding in and out)
 */
template<typename ModeType = FanAvg>
class VarianceScalingUniformInit
{
 public:
  // Empty Constructor
  // Scaling factor is the amount by which bound will be scaled.
  VarianceScalingUniformInit(const size_t scalingFactor = 1):
  scalingFactor(scalingFactor)
  {}

  template<typename eT, typename Mode = ModeType>
  void Initialize(
      arma::Mat<eT>& W,
      const size_t rows,
      const size_t cols,
      typename std::enable_if_t<std::is_same<Mode, FanIn>::value>* = 0)
  {
    double limit = sqrt(3 * scalingFactor / (double) (rows));
    RandomInitialization init(-limit, limit);
    init.Initialize(W, rows, cols);
  }

  template<typename eT, typename Mode = ModeType>
  void Initialize(
      arma::Mat<eT>& W,
      const size_t rows,
      const size_t cols,
      typename std::enable_if_t<std::is_same<Mode, FanOut>::value>* = 0)
  {
    double limit = sqrt(3 * scalingFactor / (double) (cols));
    RandomInitialization init(-limit, limit);
    init.Initialize(W, rows, cols);
  }

  template<typename eT, typename Mode = ModeType>
  void Initialize(
      arma::Mat<eT>& W,
      const size_t rows,
      const size_t cols,
      typename std::enable_if_t<std::is_same<Mode, FanAvg>::value>* = 0)
  {
    double limit = sqrt(6 * scalingFactor / 
        ((double) (rows) + (double) (cols)));
    RandomInitialization init(-limit, limit);
    init.Initialize(W, rows, cols);
  }


  /**
   * Initialize randomly the elements of the specified weight 3rd order tensor.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slice Numbers of slices.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::Cube<eT>(rows, cols, slices);

    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  const size_t scalingFactor;
}; // class VarianceScalingUniformInit


} // namespace ann
} // namespace mlpack

#endif
