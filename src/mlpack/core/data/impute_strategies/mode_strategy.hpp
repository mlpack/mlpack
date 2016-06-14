/**
 * @file mode_strategy.hpp
 * @author Keon Kim
 *
 * Definition and Implementation of the ModeStrategy class.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MODE_STRATEGY_HPP
#define MLPACK_CORE_DATA_IMPUTE_STRATEGIES_MODE_STRATEGY_HPP

#include <mlpack/core.hpp>


using namespace std;

namespace mlpack {
namespace data {

class ModeStrategy
{
 public:
  template <typename T>
  void Impute(const arma::Mat<T> &input,
              arma::Mat<T> &output,
              const size_t dimension,
              const size_t index)
  {
    // TODO: implement this
    // considering use of arma::hist()
    output(dimension, index) = 99;
    cout << "IMPUTE CALLED CUSTOM MAP STRATEGY" << endl;

  }
};

} // namespace data
} // namespace mlpack

#endif
