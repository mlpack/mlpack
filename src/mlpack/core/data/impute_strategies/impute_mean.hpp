/**
 * @file mean_strategy.hpp
 * @author Keon Kim
 *
 * Defines the DatasetInfo class, which holds information about a dataset.  This
 * is useful when the dataset contains categorical non-numeric features that
 * needs to be mapped to categorical numeric features.
 */
#ifndef MLPACK_CORE_DATA_IMPUTE_MEAN_HPP
#define MLPACK_CORE_DATA_IMPUTE_MEAN_HPP

#include <mlpack/core.hpp>


using namespace std;

namespace mlpack {
namespace data {

class ImputeMean
{
 public:
  typedef size_t impute_type_t;

  template <typename T>
  void Impute(const arma::Mat<T> &input,
              arma::Mat<T> &output,
              const size_t dimension,
              const size_t index)
  {
    output(dimension, index) = 99;
    cout << "IMPUTE CALLED MEAN MAP POLICY" << endl;

  }
};

} // namespace data
} // namespace mlpack

#endif
