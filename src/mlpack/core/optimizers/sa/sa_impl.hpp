/**
 * @file sa_impl.hpp
 * @auther Zhihao Lou
 *
 * The implementation of the SA optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SA_SA_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SA_SA_IMPL_HPP

#include <mlpack/core/dists/laplace_distribution.hpp>
#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template<typename CoolingScheduleType>
SA<CoolingScheduleType>::SA(
    CoolingScheduleType& coolingSchedule,
    const size_t maxIterations,
    const double initT,
    const size_t initMoves,
    const size_t moveCtrlSweep,
    const double tolerance,
    const size_t maxToleranceSweep,
    const double maxMoveCoef,
    const double initMoveCoef,
    const double gain) :
    coolingSchedule(coolingSchedule),
    maxIterations(maxIterations),
    temperature(initT),
    initMoves(initMoves),
    moveCtrlSweep(moveCtrlSweep),
    tolerance(tolerance),
    maxToleranceSweep(maxToleranceSweep),
    maxMoveCoef(maxMoveCoef),
    initMoveCoef(initMoveCoef),
    gain(gain)
{
  // Nothing to do.
}

//! Optimize the function (minimize).
template<typename CoolingScheduleType>
template<typename FunctionType>
double SA<CoolingScheduleType>::Optimize(FunctionType& function,
                                         arma::mat& iterate)
{
  // Make sure we have the methods that we need.
  traits::CheckNonDifferentiableFunctionTypeAPI<FunctionType>();

  const size_t rows = iterate.n_rows;
  const size_t cols = iterate.n_cols;

  size_t frozenCount = 0;
  double energy = function.Evaluate(iterate);
  double oldEnergy = energy;

  size_t idx = 0;
  size_t sweepCounter = 0;

  arma::mat accept(rows, cols, arma::fill::zeros);
  arma::mat moveSize(rows, cols);
  moveSize.fill(initMoveCoef);

  // Initial moves to get rid of dependency of initial states.
  for (size_t i = 0; i < initMoves; ++i)
    GenerateMove(function, iterate, accept, moveSize, energy, idx,
        sweepCounter);

  // Iterating and cooling.
  for (size_t i = 0; i != maxIterations; ++i)
  {
    oldEnergy = energy;
    GenerateMove(function, iterate, accept, moveSize, energy, idx,
        sweepCounter);
    temperature = coolingSchedule.NextTemperature(temperature, energy);

    // Determine if the optimization has entered (or continues to be in) a
    // frozen state.
    if (std::abs(energy - oldEnergy) < tolerance)
      ++frozenCount;
    else
      frozenCount = 0;

    // Terminate, if possible.
    if (frozenCount >= maxToleranceSweep * moveCtrlSweep * iterate.n_elem)
    {
      Log::Debug << "SA: minimized within tolerance " << tolerance << " for "
          << maxToleranceSweep << " sweeps after " << i << " iterations; "
          << "terminating optimization." << std::endl;
      return energy;
    }
  }

  Log::Debug << "SA: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;
  return energy;
}

/**
 * GenerateMove proposes a move on element iterate(idx), and determines
 * it that move is acceptable or not according to the Metropolis criterion.
 * After that it increments idx so next call will make a move on next
 * parameters. When all elements of the state have been moved (a sweep), it
 * resets idx and increments sweepCounter. When sweepCounter reaches
 * moveCtrlSweep, it performs moveControl and resets sweepCounter.
 */
template<typename CoolingScheduleType>
template<typename FunctionType>
void SA<CoolingScheduleType>::GenerateMove(
    FunctionType& function,
    arma::mat& iterate,
    arma::mat& accept,
    arma::mat& moveSize,
    double& energy,
    size_t& idx,
    size_t& sweepCounter)
{
  const double prevEnergy = energy;
  const double prevValue = iterate(idx);

  // It is possible to use a non-Laplace distribution here, but it is difficult
  // because the acceptance ratio should be as close to 0.44 as possible, and
  // MoveControl() is derived for the Laplace distribution.

  // Sample from a Laplace distribution with scale parameter moveSize(idx).
  const double unif = 2.0 * math::Random() - 1.0;
  const double move = (unif < 0) ? (moveSize(idx) * std::log(1 + unif)) :
      (-moveSize(idx) * std::log(1 - unif));

  iterate(idx) += move;
  energy = function.Evaluate(iterate);
  // According to the Metropolis criterion, accept the move with probability
  // min{1, exp(-(E_new - E_old) / T)}.
  const double xi = math::Random();
  const double delta = energy - prevEnergy;
  const double criterion = std::exp(-delta / temperature);
  if (delta <= 0. || criterion > xi)
  {
    accept(idx) += 1.;
  }
  else // Reject the move; restore previous state.
  {
    iterate(idx) = prevValue;
    energy = prevEnergy;
  }

  ++idx;
  if (idx == iterate.n_elem) // Finished with a sweep.
  {
    idx = 0;
    ++sweepCounter;
  }

  if (sweepCounter == moveCtrlSweep) // Do MoveControl().
  {
    MoveControl(moveCtrlSweep, accept, moveSize);
    sweepCounter = 0;
  }
}

/**
 * MoveControl() uses a proportional feedback control to determine the size
 * parameter to pass to the move generation distribution. The target of such
 * move control is to make the acceptance ratio, accept/nMoves, be as close to
 * 0.44 as possible. Generally speaking, the larger the move size is, the larger
 * the function value change of the move will be, and less likely such move will
 * be accepted by the Metropolis criterion. Thus, the move size is controlled by
 *
 * log(moveSize) = log(moveSize) + gain * (accept/nMoves - target)
 *
 * For more theory and the mysterious 0.44 value, see Jimmy K.-C. Lam and
 * Jean-Marc Delosme. `An efficient simulated annealing schedule: derivation'.
 * Technical Report 8816, Yale University, 1988.
 */
template<typename CoolingScheduleType>
inline void SA<CoolingScheduleType>::MoveControl(const size_t nMoves,
                                                 arma::mat& accept,
                                                 arma::mat& moveSize)
{
  arma::mat target;
  target.copy_size(accept);
  target.fill(0.44);
  moveSize = arma::log(moveSize);
  moveSize += gain * (accept / (double) nMoves - target);
  moveSize = arma::exp(moveSize);

  // To avoid the use of element-wise arma::min(), which is only available in
  // Armadillo after v3.930, we use a for loop here instead.
  for (size_t i = 0; i < accept.n_elem; ++i)
    moveSize(i) = (moveSize(i) > maxMoveCoef) ? maxMoveCoef : moveSize(i);

  accept.zeros();
}

} // namespace optimization
} // namespace mlpack

#endif
