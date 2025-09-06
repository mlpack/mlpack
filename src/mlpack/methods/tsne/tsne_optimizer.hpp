/**
 * @file methods/tsne/tsne_optimizer.hpp
 * @author Ranjodh Singh
 *
 * The Default TSNE Optimizer. (Implemented as a UpdatePolicy for ens::SGD)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_OPTIMIZER_HPP
#define MLPACK_METHODS_TSNE_TSNE_OPTIMIZER_HPP

#include <ensmallen.hpp>
#include <mlpack/prereqs.hpp>

namespace mlpack
{

class DeltaBarDeltaUpdate
{
 public:
  /**
   * Construct the DeltaBarDelta update policy with given parameters.
   *
   * @param beta1 The beta1 hyperparameter
   * @param beta2 The beta2 hyperparameter
   * @param momentum The momentup hyperparameter
   * @param minGain The minGain hyperparameter
   */
  DeltaBarDeltaUpdate(const double beta1 = 0.2,
                      const double beta2 = 0.8,
                      const double momentum = 0.5,
                      const double minGain = 0.01)
      : beta1(beta1), beta2(beta2), momentum(momentum),
        minGain(minGain) { /* Do nothing. */ };

  //! Access beta1.
  double Beta1() const { return beta1; }
  //! Modify beta1.
  double& Beta1() { return beta1; }

  //! Access beta2.
  double Beta2() const { return beta2; }
  //! Modify beta2.
  double& Beta2() { return beta2; }

  //! Access the momentum.
  double Momentum() const { return momentum; }
  //! Modify the momentum.
  double& Momentum() { return momentum; }

  //! Access the minGain
  double MinimumGain() const { return minGain; }
  //! Modify the minGain.
  double& MinimumGain() { return minGain; }

  /**
   * The UpdatePolicyType policy classes must contain an internal 'Policy'
   * template class with two template arguments: MatType and GradType.  This is
   * instantiated at the start of the optimization, and holds parameters
   * specific to an individual optimization.
   */
  template <typename MatType, typename GradType>
  class Policy
  {
   public:
    /**
     * This is called by the optimizer method before the start of the iteration
     * update process.
     *
     * @param parent Instantiated parent class.
     * @param rows Number of rows in the gradient matrix.
     * @param cols Number of columns in the gradient matrix.
     */
    Policy(const DeltaBarDeltaUpdate& parent,
           const size_t rows,
           const size_t cols)
        : parent(parent), velocity(rows, cols)
    {
      gains.ones(rows, cols);
    }

    /**
     * Update step for SGD.
     *
     * @param iterate Parameters that minimize the function.
     * @param stepSize Step size to be used for the given iteration.
     * @param gradient The gradient matrix.
     */
    void Update(MatType& iterate,
                const double stepSize,
                const GradType& gradient)
    {
      arma::umat increase = (velocity % gradient) < 0.0;
      arma::umat decrease = (velocity % gradient) > 0.0;
      gains.elem(arma::find(increase)) += parent.Beta1();
      gains.elem(arma::find(decrease)) *= parent.Beta2();
      gains.elem(arma::find(gains < parent.MinimumGain()))
          .fill(parent.MinimumGain());

      velocity = parent.momentum * velocity - stepSize * (gains % gradient);
      iterate += velocity;
    }

   private:
    // The instantiated parent class.
    const DeltaBarDeltaUpdate& parent;
    // The gains matrix.
    MatType gains;
    // The velocity matrix.
    MatType velocity;
  };

 private:
  // The beta1 hyperparameter
  double beta1;
  // The beta2 hyperparameter
  double beta2;
  // The momentum hyperparameter.
  double momentum;
  // The minGain hyperparameter
  double minGain;
};

using TSNEOptimizer = ens::SGD<DeltaBarDeltaUpdate>;

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_OPTIMIZER_HPP
