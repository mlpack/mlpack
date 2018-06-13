/**
 * @file dual_optimizer.hpp
 * @author Shikhar Jaiswal
 *
 * Dual optimizer class. This class is used as a workaround for training
 * Generative Adversarial Networks in mlpack using separate optimizers for
 * the generator and the discriminator, without breaking the single-optimizer
 * API of the library.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_DUAL_OPTIMIZER_DUAL_OPTIMIZER_HPP
#define MLPACK_CORE_OPTIMIZERS_DUAL_OPTIMIZER_DUAL_OPTIMIZER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 */
template<class DiscriminatorOptimizer, class GeneratorOptimizer>
class DualOptimizer
{
 public:
  /**
   */
  DualOptimizer(DiscriminatorOptimizer& optDiscriminator,
               GeneratorOptimizer& optGenerator);

  /**
   */
  template<typename DecomposableFunctionTypeDiscriminator,
           typename DecomposableFunctionTypeGenerator>
  void Optimize(DecomposableFunctionTypeDiscriminator& functionDiscriminator,
                DecomposableFunctionTypeGenerator& functionGenerator,
                arma::mat& iterate)
  {
    optDiscriminator.Optimize(functionDiscriminator, iterate);
    optGenerator.Optimize(functionGenerator, iterate);
  }

  //! Get the discriminator step size.
  double DiscriminatorStepSize() const
  {
    return optDiscriminator.StepSize();
  }
  //! Modify the discriminator step size.
  double& DiscriminatorStepSize()
  {
    return optDiscriminator.StepSize();
  }

  //! Get the generator step size.
  double GeneratorStepSize() const
  {
    return optGenerator.StepSize();
  }
  //! Modify the generator step size.
  double& GeneratorStepSize()
  {
    return optGenerator.StepSize();
  }

  //! Get the discriminator batch size.
  size_t DiscriminatorBatchSize() const
  {
    return optDiscriminator.BatchSize();
  }
  //! Modify the discriminator batch size.
  size_t& DiscriminatorBatchSize()
  {
    return optDiscriminator.BatchSize();
  }

  //! Get the generator batch size.
  size_t GeneratorBatchSize() const
  {
    return optGenerator.BatchSize();
  }
  //! Modify the generator batch size.
  size_t& GeneratorBatchSize()
  {
    return optGenerator.BatchSize();
  }

  //! Get the value used to initialise the mean squared gradient parameter
  //! in discriminator.
  double DiscriminatorEpsilon() const
  {
    return optDiscriminator.UpdatePolicy().Epsilon();
  }
  //! Modify the value used to initialise the mean squared gradient parameter
  //! in discriminator.
  double& DiscriminatorEpsilon()
  {
    return optDiscriminator.UpdatePolicy().Epsilon();
  }

  //! Get the value used to initialise the mean squared gradient parameter
  //! in generator.
  double GeneratorEpsilon() const
  {
    return optGenerator.UpdatePolicy().Epsilon();
  }
  //! Modify the value used to initialise the mean squared gradient parameter
  //! in generator.
  double& GeneratorEpsilon()
  {
    return optGenerator.UpdatePolicy().Epsilon();
  }

  //! Get the maximum number of iterations (0 indicates no limit)
  //! in discriminator.
  size_t DiscriminatorMaxIterations() const
  {
    return optDiscriminator.MaxIterations();
  }
  //! Modify the maximum number of iterations (0 indicates no limit)
  //! in discriminator.
  size_t& DiscriminatorMaxIterations()
  {
    return optDiscriminator.MaxIterations();
  }

  //! Get the maximum number of iterations (0 indicates no limit)
  //! in generator.
  size_t GeneratorMaxIterations() const
  {
    return optGenerator.MaxIterations();
  }
  //! Modify the maximum number of iterations (0 indicates no limit)
  //! in generator.
  size_t& GeneratorMaxIterations()
  {
    return optGenerator.MaxIterations();
  }

  //! Get the tolerance for termination in discriminator.
  double DiscriminatorTolerance() const
  {
    return optDiscriminator.Tolerance();
  }
  //! Modify the tolerance for termination in discriminator.
  double& DiscriminatorTolerance()
  {
    return optDiscriminator.Tolerance();
  }

  //! Get the tolerance for termination in generator.
  double GeneratorTolerance() const
  {
    return optGenerator.Tolerance();
  }
  //! Modify the tolerance for termination in generator.
  double& GeneratorTolerance()
  {
    return optGenerator.Tolerance();
  }

  //! Get whether or not the individual functions are shuffled
  //! in discriminator.
  bool DiscriminatorShuffle() const { return optDiscriminator.Shuffle(); }
  //! Modify whether or not the individual functions are shuffled
  //! in discriminator.
  bool& DiscriminatorShuffle() { return optDiscriminator.Shuffle(); }

  //! Get whether or not the individual functions are shuffled in generator.
  bool GeneratorShuffle() const { return optGenerator.Shuffle(); }
  //! Modify whether or not the individual functions are shuffled in generator.
  bool& GeneratorShuffle() { return optGenerator.Shuffle(); }

 private:
  //! Locally stored Generator optimizer.
  GeneratorOptimizer& optGenerator;
  //! Locally stored Discriminator optimizer.
  DiscriminatorOptimizer& optDiscriminator;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "dual_optimizer_impl.hpp"

#endif
