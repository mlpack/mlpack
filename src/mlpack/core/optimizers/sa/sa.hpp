/**
 * @file sa.hpp
 * @author Zhihao Lou
 *
 * Simulated Annealing (SA).
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SA_SA_HPP
#define __MLPACK_CORE_OPTIMIZERS_SA_SA_HPP

namespace mlpack {
namespace optimization {

/**
 * Simulated Annealing is an stochastic optimization algorithm which is able to
 * deliver near-optimal results quickly without knowing the gradient of the
 * function being optimized. It has unique hill climbing capability that makes
 * it less vulnerable to local minima.  This implementation uses exponential
 * cooling schedule and feedback move control by default, but the cooling
 * schedule can be changed via a template parameter.
 *
 * The algorithm keeps the temperature at initial temperature for initMove
 * steps to get rid of the dependency of initial condition. After that, it
 * cools every step until the system is considered frozen or maxIterations is
 * reached.
 *
 * At each step, SA only perturbs one parameter at a time. The process that SA
 * perturbed all parameters in a problem is called a sweep. Every moveCtrlSweep
 * the algorithm does feedback move control to change the average move size
 * depending on the responsiveness of each parameter. Parameter gain controls
 * the proportion of the feedback control.
 *
 * The system is considered "frozen" when its score failed to change more then
 * tolerance for consecutive maxToleranceSweep sweeps.
 *
 * For SA to work, the FunctionType parameter must implement the following
 * two methods:
 *
 *   double Evaluate(const arma::mat& coordinates);
 *   arma::mat& GetInitialPoint();
 *
 * and the CoolingScheduleType parameter must implement the following method:
 *
 *   double NextTemperature(const double currentTemperature,
 *                          const double currentValue);
 *
 * which returns the next temperature given current temperature and the value
 * of the function being optimized.
 *
 * @tparam FunctionType objective function type to be minimized.
 * @tparam CoolingScheduleType type for cooling schedule
 */
template<typename FunctionType, typename CoolingScheduleType>
class SA
{
 public:
  /*
   * Construct the SA optimizer with the given function and parameters.
   *
   * @param function Function to be minimized.
   * @param coolingSchedule Instantiated cooling schedule.
   * @param maxIterations Maximum number of iterations allowed (0 indicates no limit).
   * @param initT Initial temperature.
   * @param initMoves Number of initial iterations without changing temperature.
   * @param moveCtrlSweep Sweeps per feedback move control.
   * @param tolerance Tolerance to consider system frozen.
   * @param maxToleranceSweep Maximum sweeps below tolerance to consider system
   *      frozen.
   * @param maxMoveCoef Maximum move size.
   * @param initMoveCoef Initial move size.
   * @param gain Proportional control in feedback move control.
   */
  SA(FunctionType& function,
     CoolingScheduleType& coolingSchedule,
     const size_t maxIterations = 1000000,
     const double initT = 10000.,
     const size_t initMoves = 1000,
     const size_t moveCtrlSweep = 100,
     const double tolerance = 1e-5,
     const size_t maxToleranceSweep = 3,
     const double maxMoveCoef = 20,
     const double initMoveCoef = 0.3,
     const double gain = 0.3);


  /*&
   * Optimize the given function using simulated annealing. The given starting
   * point will be modified to store the finishing point of the algorithm, and
   * the final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate);

  //! Get the instantiated function to be optimized.
  const FunctionType& Function() const { return function; }
  //! Modify the instantiated function.
  FunctionType& Function() { return function; }

  //! Get the temperature.
  double Temperature() const { return temperature; }
  //! Modify the temperature.
  double& Temperature() { return temperature; }

  //! Get the initial moves.
  size_t InitMoves() const { return initMoves; }
  //! Modify the initial moves.
  size_t& InitMoves() { return initMoves; }

  //! Get sweeps per move control.
  size_t MoveCtrlSweep() const { return moveCtrlSweep; }
  //! Modify sweeps per move control.
  size_t& MoveCtrlSweep() { return moveCtrlSweep; }

  //! Get the tolerance.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance.
  double& Tolerance() { return tolerance; }

  //! Get the maxToleranceSweep.
  size_t MaxToleranceSweep() const { return maxToleranceSweep; }
  //! Modify the maxToleranceSweep.
  size_t& MaxToleranceSweep() { return maxToleranceSweep; }

  //! Get the gain.
  double Gain() const { return gain; }
  //! Modify the gain.
  double& Gain() { return gain; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the maximum move size of each parameter.
  arma::mat MaxMove() const { return maxMove; }
  //! Modify the maximum move size of each parameter.
  arma::mat& MaxMove() { return maxMove; }

  //! Get move size of each parameter.
  arma::mat MoveSize() const { return moveSize; }
  //! Modify move size of each parameter.
  arma::mat& MoveSize() { return moveSize; }

  //! Return a string representation of this object.
  std::string ToString() const;
 private:
  //! The function to be optimized.
  FunctionType& function;
  //! The cooling schedule being used.
  CoolingScheduleType& coolingSchedule;
  //! The maximum number of iterations.
  size_t maxIterations;
  //! The current temperature.
  double temperature;
  //! The number of initial moves before reducing the temperature.
  size_t initMoves;
  //! The number of sweeps before a MoveControl() call.
  size_t moveCtrlSweep;
  //! Tolerance for convergence.
  double tolerance;
  //! Number of sweeps in tolerance before system is considered frozen.
  size_t maxToleranceSweep;
  //! Proportional control in feedback move control.
  double gain;

  //! Maximum move size of each parameter.
  arma::mat maxMove;
  //! Move size of each parameter.
  arma::mat moveSize;


  void GenerateMove(arma::mat& iterate,
                    arma::mat& accept,
                    double& energy,
                    size_t& idx,
                    size_t& sweepCounter);

  void MoveControl(size_t nMoves, arma::mat& accept);
};

}; // namespace optimization
}; // namespace mlpack

#include "sa_impl.hpp"

#endif
