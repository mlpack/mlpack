#ifndef __MLPACK_CORE_UTIL_SPLIT_DATA_HPP
#define __MLPACK_CORE_UTIL_SPLIT_DATA_HPP

#include <mlpack/core.hpp>

#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

namespace mlpack {
namespace util {

/**
 *Split training data and test data, please define
 *ARMA_USE_CXX11 to enable move of c++11
 */
class TrainTestSplit
{
public:
  /**
   * @brief TrainTestSplit
   * @param testRatio the ratio of test data
   * @param seed seed of the random device
   * @warning slice should not less than 1
   */
  TrainTestSplit(double testRatio,                
                 arma::arma_rng::seed_type seed = 0) :
    seed(seed),    
    testRatio(testRatio)
  {    
  }

  /**
   *Split training data and test data, please define
   *ARMA_USE_CXX11 to enable move of c++11
   *@param input input data want to split
   *@param label input label want to split
   *@param testRatio the ratio of test data
   *@param seed seed of the random device
   *@code
   *arma::mat input = loadData();
   *arma::Row<size_t> label = loadLabel();
   *arma::mat trainData;
   *arma::mat testData;
   *arma::Row<size_t> trainLabel;
   *arma::Row<size_t> testLabel;
   *std::random_device rd;
   *TrainTestSplit tts(0.25);
   *tts.Split(input, label, trainData, testData, trainLabel,
   *          testLabel);
   *@endcode
   */
  template<typename T, typename U>
  void Split(arma::Mat<T> const &input,
             arma::Row<U> const &inputLabel,
             arma::Mat<T> &trainData,
             arma::Mat<T> &testData,
             arma::Row<U> &trainLabel,
             arma::Row<U> &testLabel)
  {
    size_t const testSize =
        static_cast<size_t>(input.n_cols * testRatio);
    size_t const trainSize = input.n_cols - testSize;
    trainData.set_size(input.n_rows, trainSize);
    testData.set_size(input.n_rows, testSize);
    trainLabel.set_size(trainSize);
    testLabel.set_size(testSize);

    using Col = arma::Col<size_t>;
    arma::arma_rng::set_seed(seed);
    Col const sequence = arma::linspace<Col>(0, input.n_cols - 1,
                                             input.n_cols);
    arma::Col<size_t> const order = arma::shuffle(sequence);

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
   */
  template<typename T,typename U>
  std::tuple<arma::Mat<T>, arma::Mat<T>,
  arma::Row<U>, arma::Row<U>>
  Split(arma::Mat<T> const &input,
        arma::Row<U> const &inputLabel)
  {
    arma::Mat<T> trainData;
    arma::Mat<T> testData;
    arma::Row<U> trainLabel;
    arma::Row<U> testLabel;

    Split(input, inputLabel, trainData, testData,
          trainLabel, testLabel);

    return std::make_tuple(trainData, testData,
                           trainLabel, testLabel);
  }

  void Seed(arma::arma_rng::seed_type value)
  {
    seed = value;
  }
  arma::arma_rng::seed_type Seed() const
  {
    return seed;
  }

  size_t Slice() const
  {
    return slice;
  }
  void Slice(size_t value)
  {
    if(value < 1){
      throw std::out_of_range("The range of slice should not less than 1");
    }
    slice = value;
  }

  void TestRatio(double value)
  {
    testRatio = value;
  }
  double TestRatio() const
  {
    return testRatio;
  }

private:  
  arma::arma_rng::seed_type seed;
  size_t slice;
  double testRatio;
};

} // namespace util
} // namespace mlpack

#endif
