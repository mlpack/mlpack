/**
 * @file variance_scaling_init.hpp
 * @author Shashank Shekhar
 *
 * Intialization rule for the neural networks.
 *
 * VarianceScalingNormalInit is performed by assigning a gaussian matrix with
 * zero mean and variance given by (scaling factor / N) to the weight matrix.
 *
 * VarianceScalingUniformInit is performed by assigning a random matrix between
 * [-limit, limit] where limit is given as sqrt(3*scaling facor/N).
 *
 * N is decided by the mode of initializer and is described below.
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

/**
 * This class defines the mode of initialization as being Fan-In. This mode is
 * defined by :
 * @f$ N &=& n_i @f$
 *
 * where @f$ n_i @f$ represents the number of neurons in the ingoing layer.
 */
class FanIn
{};

/**
 * This class defines the mode of initialization as being Fan-Out. This mode is
 * defined by :
 * @f$ N &=& n_{i+1} @f$
 *
 * where @f$ n_{i+1} @f$ represents the number of neurons in the outgoing layer.
 */
class FanOut
{};

/**
 * This class defines the mode of initialization as being Fan-In. This mode is
 * defined by :
 * @f$ N &=& \frac{2}{n_i + n_{i+1}}
 *
 * where @f$ n_{i+1} @f$ is the number of neurons in the outgoing layer, 
 * @f$ n_i @f$ represents the number of neurons in the ingoing layer.
 */
class FanAvg
{};


/**
 * This class is used to initialize weigth matrix from a gaussian distribution
 * with zero mean and variance decided by mode type. The method is defined by
 * 
 * @f{eqnarray*}{
 * \mathrm{Var}[w_i] &=& \frac{scalingFactor}{N} \\
 *
 * w_i \sim \mathrm{N}(0,\mathrm{Var}[w_i])
 * @f}
 *
 * where @f$ N @f$ is decided by the mode type of the initialization. 
 * There are three modes of initialization as defined by :
 * 
 * <b> Fan-In Mode : </b> @f$ N &=& n_i @f$
 * <b> Fan-Out Mode : </b> @f$ N &=& n_{i+1} @f$
 * <b> Fan-Avg Mode : </b> @f$ N &=& \frac{2}{n_i + n_{i+1}} @f$
 *
 * where @f$ n_{i+1} @f$ is the number of neurons in the outgoing layer, 
 * @f$ n_i @f$ represents the number of neurons in the ingoing layer.
 *
 */
template<typename ModeType = FanAvg>
class VarianceScalingNormalInit
{
 public:
  /**
  * Initialize the Variance Scaling Normal initializers.
  *
  * @param scalingFactor Number by which to multiply variance.
  */
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
    double var = (scalingFactor / double(rows));
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
    double var = (scalingFactor / double(cols));
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
    double var = (2 * scalingFactor / (double(rows + cols)));
    GaussianInitialization init(0, var);
    init.Initialize(W, rows, cols);
  }


  /**
   * Initialize the elements of the specified weight 3rd order tensor with
   * variance scaling normal initialization method.
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
 * This class is used to initialize weigth matrix from a uniform distribution
 * with zero mean and range decided by mode type. The method is defined by :
 * 
 * @f{eqnarray*}{
 * \mathrm{Var}[w_i] &=& \frac{scaling factor}{N} \\
 *
 * w_i \sim \mathrm{U}[-\frac{\sqrt{3 &times; scalingFactor}}{\sqrt{N}},
 * \frac{\sqrt{3 &times; scalingFactor}}{\sqrt{N}}]
 * @f}
 *
 * where @f$ N @f$ is decided by the mode type of the initialization. 
 * There are three modes of initialization as defined by :
 *
 * <b> Fan-In Mode : </b> @f$ N &=& n_i @f$
 * <b> Fan-Out Mode : </b> @f$ N &=& n_{i+1} @f$
 * <b> Fan-Avg Mode : </b> @f$ N &=& \frac{2}{n_i + n_{i+1}} @f$
 *
 * where @f$ n_{i+1} @f$ is the number of neurons in the outgoing layer, 
 * @f$ n_i @f$ represents the number of neurons in the ingoing layer.
 *
 */
template<typename ModeType = FanAvg>
class VarianceScalingUniformInit
{
 public:
  /**
  * Initialize the Variance Scaling Uniform initializers.
  *
  * @param scalingFactor Number by which to multiply range.
  */
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
    double limit = sqrt(3 * scalingFactor / double(rows));
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
    double limit = sqrt(3 * scalingFactor / double(cols));
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
    double limit = sqrt(6 * scalingFactor / (double(rows + cols)));
    RandomInitialization init(-limit, limit);
    init.Initialize(W, rows, cols);
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with
   * variance scaling uniform initialization method.
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
