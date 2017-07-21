#ifndef MLPACK_CORE_OPTIMIZERS_SGD_EMPTY_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_EMPTY_UPDATE_HPP

#include <mlpack/prereqs.hpp>

template<typename UpdatePolicy>
class GradientClipping
{
 public:
  GradientClipping(const double minGradient,
                   const double maxGradient,
                   UpdatePolicy updatePolicy) :
    minGradient(minGradient),
    maxGradient(maxGradient),
    updatePolicy(updatePolicy)
  {
    // Nothing to do here
  }

  void Initialize(const size_t /* rows */, const size_t /* cols */)
  { /* Do nothing. */ }

  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    gradient.transform
    (
      [&](double val)
      {
        return std::min(std::max(val, minGradient), maxGradient);
      }
    );    
    updatePolicy.Update(iterate, stepSize, gradient);
  }
 private:
  double minGradient;
  double maxGradient;
  UpdatePolicy updatePolicy;
};

#endif
