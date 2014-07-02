/*
 * @file laplace.hpp
 * @author Zhihao Lou
 *
 * Laplace (double exponential) distribution used in SA
 */

#ifndef __MLPACK_CORE_OPTIMIZER_SA_LAPLACE_DISTRIBUTION_HPP
#define __MLPACK_CORE_OPTIMIZER_SA_LAPLACE_DISTRIBUTION_HPP

namespace mlpack {
namespace optimization {

/* 
 * The Laplace distribution centered at 0 has pdf
 * \f[
 * f(x|\theta) = \frac{1}{2\theta}\exp\left(-\frac{|x|}{\theta}\right)
 * \f]
 * given scale parameter \f$\theta\f$.
 */
class LaplaceDistribution
{
 public:
  //! Nothing to do for the constructor
  LaplaceDistribution(){}
  //! Return random value from Laplace distribution with parameter param
  double operator () (const double param);

};

}; // namespace optimization
}; // namespace mlpack

#endif
