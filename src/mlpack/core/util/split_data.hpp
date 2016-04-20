#ifndef __MLPACK_CORE_UTIL_SPLIT_DATA_HPP
#define __MLPACK_CORE_UTIL_SPLIT_DATA_HPP

#include <mlpack/core.hpp>

#include <tuple>

namespace mlpack {
namespace util {

/**
 *Split training data and test data
 *@param input input data want to split
 *@param label input label want to split
 *@param trainData training data split by input
 *@param testData test data split by input
 *@param trainLabel train label split by input
 *@param testLabel test label split by input
 *@param testRatio the ratio of test data
 *@code
 *arma::mat input = loadData();
 *arma::Row<size_t> label = loadLabel();
 *arma::mat trainData;
 *arma::mat testData;
 *arma::Row<size_t> trainLabel;
 *arma::Row<size_t> testLabel;
 *arma::arma_rng::set_seed(100); //set the seed if you like
 *TrainTestSplit(input, label, trainData,
 *               testData, trainLabel, testLabel);
 *@endcode
 */
template<typename T, typename U>
void TrainTestSplit(const arma::Mat<T> &input,
                    const arma::Row<U> &inputLabel,
                    arma::Mat<T> &trainData,
                    arma::Mat<T> &testData,
                    arma::Row<U> &trainLabel,
                    arma::Row<U> &testLabel,
                    const double testRatio)
{
  size_t const testSize =
      static_cast<size_t>(input.n_cols * testRatio);
  const size_t trainSize = input.n_cols - testSize;
  trainData.set_size(input.n_rows, trainSize);
  testData.set_size(input.n_rows, testSize);
  trainLabel.set_size(trainSize);
  testLabel.set_size(testSize);

  const arma::Col<size_t> order =
      arma::shuffle(arma::linspace<arma::Col<size_t>>(0, input.n_cols - 1,
                                                      input.n_cols));

  for(size_t i = 0; i != trainSize; ++i)
  {
    trainData.col(i) = input.col(order[i]);
    trainLabel(i) = inputLabel(order[i]);
  }

  for(size_t i = 0; i != testSize; ++i)
  {
    testData.col(i) = input.col(order[i + trainSize]);
    testLabel(i) = inputLabel(order[i + trainSize]);
  }
}

/**
 *Overload of Split, if you do not like to pass in
 *so many param, you could call this api instead
 *@param input input data want to split
 *@param label input label want to split
 *@return They are trainData, testData, trainLabel and
 *testLabel
 *@code
 *arma::mat input = loadData();
 *arma::Row<size_t> label = loadLabel();
 *auto splitResult = TrainTestSplit(input, label, 0.2);
 *@endcode
 */
template<typename T,typename U>
std::tuple<arma::Mat<T>, arma::Mat<T>,
arma::Row<U>, arma::Row<U>>
TrainTestSplit(const arma::Mat<T> &input,
               const arma::Row<U> &inputLabel,
               const double testRatio)
{
  arma::Mat<T> trainData;
  arma::Mat<T> testData;
  arma::Row<U> trainLabel;
  arma::Row<U> testLabel;

  TrainTestSplit(input, inputLabel,
                 trainData, testData,
                 trainLabel, testLabel,
                 testRatio);

  return std::make_tuple(trainData, testData,
                         trainLabel, testLabel);
}

} // namespace util
} // namespace mlpack

#endif
