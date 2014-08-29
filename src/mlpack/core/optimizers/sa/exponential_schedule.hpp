/**
 * @file exponential_schedule.hpp
 * @author Zhihao Lou
 *
 * Exponential (geometric) cooling schedule used in SA.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SA_EXPONENTIAL_SCHEDULE_HPP
#define __MLPACK_CORE_OPTIMIZERS_SA_EXPONENTIAL_SCHEDULE_HPP

namespace mlpack {
namespace optimization {

/**
 * The exponential cooling schedule cools the temperature T at every step
 * according to the equation
 *
 * \f[
 * T_{n+1} = (1-\lambda) T_{n}
 * \f]
 *
 * where \f$ 0<\lambda<1 \f$ is the cooling speed. The smaller \f$ \lambda \f$
 * is, the slower the cooling speed, and better the final result will be. Some
 * literature uses \f$ \alpha = (-1 \lambda) \f$ instead. In practice,
 * \f$ \alpha \f$ is very close to 1 and will be awkward to input (e.g.
 * alpha = 0.999999 vs lambda = 1e-6).
 */
class ExponentialSchedule
{
 public:
  /*
   * Construct the ExponentialSchedule with the given parameter.
   *
   * @param lambda Cooling speed.
   */
  ExponentialSchedule(const double lambda = 0.001) : lambda(lambda) { }

  /**
   * Returns the next temperature given current status.  The current system's
   * energy is not used in this calculation.
   *
   * @param currentTemperature Current temperature of system.
   * @param currentEnergy Current energy of system (not used).
   */
  double NextTemperature(
      const double currentTemperature,
      const double /* currentEnergy */)
  {
    return (1 - lambda) * currentTemperature;
  }

  //! Get the cooling speed, lambda.
  double Lambda() const { return lambda; }
  //! Modify the cooling speed, lambda.
  double& Lambda() { return lambda; }

 private:
  //! The cooling speed.
  double lambda;
};

}; // namespace optimization
}; // namespace mlpack

#endif
