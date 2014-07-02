/*
 * @file sa_impl.hpp
 * @auther Zhihao Lou
 *
 * The implementation of the SA optimizer.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SA_SA_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SA_SA_IMPL_HPP

namespace mlpack {
namespace optimization {

template<
    typename FunctionType,
    typename MoveDistributionType,
    typename CoolingScheduleType
>
SA<FunctionType, MoveDistributionType, CoolingScheduleType>::SA(
    FunctionType& function,
    MoveDistributionType& moveDistribution,
    CoolingScheduleType& coolingSchedule,
    const double initT,
    const size_t initMoves,
    const size_t moveCtrlSweep,
    const double tolerance,
    const size_t maxToleranceSweep,
    const double maxMoveCoef,
    const double initMoveCoef,
    const double gain,
    const size_t maxIterations) :
    function(function),
    moveDistribution(moveDistribution),
    coolingSchedule(coolingSchedule),
    T(initT),
    initMoves(initMoves),
    moveCtrlSweep(moveCtrlSweep),
    tolerance(tolerance),
    maxToleranceSweep(maxToleranceSweep),
    gain(gain),
    maxIterations(maxIterations)
{
  const size_t rows = function.GetInitialPoint().n_rows;
  const size_t cols = function.GetInitialPoint().n_cols;

  maxMove.set_size(rows, cols);
  maxMove.fill(maxMoveCoef);
  moveSize.set_size(rows, cols);
  moveSize.fill(initMoveCoef);
  accept.zeros(rows, cols);
}

//! Optimize the function (minimize).
template<
    typename FunctionType,
    typename MoveDistributionType,
    typename CoolingScheduleType
>
double SA<FunctionType, MoveDistributionType, CoolingScheduleType>::Optimize(
    arma::mat &iterate)
{
  const size_t rows = function.GetInitialPoint().n_rows;
  const size_t cols = function.GetInitialPoint().n_cols;

  size_t i;
  size_t frozenCount = 0;
  energy = function.Evaluate(iterate);
  size_t oldEnergy = energy;
  math::RandomSeed(std::time(NULL));

  nVars = rows * cols;
  idx = 0;
  sweepCounter = 0;
  accept.zeros();

  // Initial Moves to get rid of dependency of initial states.
  for (i = 0; i < initMoves; ++i)
    GenerateMove(iterate);

  // Iterating and cooling.
  for (i = 0; i != maxIterations; ++i)
  {
    oldEnergy = energy;
    GenerateMove(iterate);
    T = coolingSchedule.nextTemperature(T, energy);

    // Determine if the optimization has entered (or continues to be in) a
    // frozen state.
    if (std::abs(energy - oldEnergy) < tolerance)
      ++frozenCount;
    else
      frozenCount = 0;

    // Terminate, if possible.
    if (frozenCount >= maxToleranceSweep * nVars)
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
 * parameters. When all elements of the state has been moved (a sweep), it
 * resets idx and increments sweepCounter. When sweepCounter reaches
 * moveCtrlSweep, it performs moveControl and resets sweepCounter.
 */
template<
    typename FunctionType,
    typename MoveDistributionType,
    typename CoolingScheduleType
>
void SA<FunctionType, MoveDistributionType, CoolingScheduleType>::GenerateMove(
    arma::mat& iterate)
{
  double prevEnergy = energy;
  double prevValue = iterate(idx);
  double move = moveDistribution(moveSize(idx));
  iterate(idx) += move;
  energy = function.Evaluate(iterate);
  // According to Metropolis criterion, accept the move with probability
  // min{1, exp(-(E_new - E_old) / T)}.
  double xi = math::Random();
  double delta = energy - prevEnergy;
  double criterion = std::exp(-delta / T);
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
  if (idx == nVars) // Finished with a sweep.
  {
    idx = 0;
    ++sweepCounter;
  }

  if (sweepCounter == moveCtrlSweep) // Do MoveControl().
  {
    MoveControl(moveCtrlSweep);
    sweepCounter = 0;
  }
}

/*
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
 * Technical Report 8816, Yale University, 1988
 */
template<
    typename FunctionType,
    typename MoveDistributionType,
    typename CoolingScheduleType
>
void SA<FunctionType, MoveDistributionType, CoolingScheduleType>::MoveControl(
    size_t nMoves)
{
  arma::mat target;
  target.copy_size(accept);
  target.fill(0.44);
  moveSize = arma::log(moveSize);
  moveSize += gain * (accept / (double) nMoves - target);
  moveSize = arma::exp(moveSize);

  // To avoid the use of element-wise arma::min(), which is only available in
  // Armadillo after v3.930, we use a for loop here instead.
  for (size_t i = 0; i < nVars; ++i)
    moveSize(i) = (moveSize(i) > maxMove(i)) ? maxMove(i) : moveSize(i);

  accept.zeros();
}

template<
    typename FunctionType,
    typename MoveDistributionType,
    typename CoolingScheduleType
>
std::string SA<FunctionType, MoveDistributionType, CoolingScheduleType>::
ToString() const
{
  std::ostringstream convert;
  convert << "SA [" << this << "]" << std::endl;
  convert << "  Function:" << std::endl;
  convert << util::Indent(function.ToString(), 2);
  convert << "  Move Distribution:" << std::endl;
  convert << util::Indent(moveDistribution.ToString(), 2);
  convert << "  Cooling Schedule:" << std::endl;
  convert << util::Indent(coolingSchedule.ToString(), 2);
  convert << "  Temperature: " << T << std::endl;
  convert << "  Initial moves: " << initMoves << std::endl;
  convert << "  Sweeps per move control: " << moveCtrlSweep << std::endl;
  convert << "  Tolerance: " << tolerance << std::endl;
  convert << "  Maximum sweeps below tolerance: " << maxToleranceSweep
      << std::endl;
  convert << "  Move control gain: " << gain << std::endl;
  convert << "  Maximum iterations: " << maxIterations << std::endl;
  return convert.str();
}

}; // namespace optimization
}; // namespace mlpack

#endif
