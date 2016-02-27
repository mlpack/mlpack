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
   * @param slice indicate how many slice(depth) per image,
   * this parameter only work on arma::Cube<T>
   * @param seed seed of the random device
   * @warning slice should not less than 1
   */
  TrainTestSplit(double testRatio,
                 size_t slice = 1,
                 arma::arma_rng::seed_type seed = 0) :
    seed(seed),
    slice(slice),
    testRatio(testRatio)
  {
    if(slice < 1){
      throw std::out_of_range("The range of slice should not less than 1");
    }
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
  template<typename T>
  void Split(T const &input,
             arma::Row<size_t> const &inputLabel,
             T &trainData,
             T &testData,
             arma::Row<size_t> &trainLabel,
             arma::Row<size_t> &testLabel)
  {
    size_t const testSize =
        static_cast<size_t>(ExtractSize(input) * testRatio);
    size_t const trainSize = ExtractSize(input) - testSize;

    ResizeData(input, trainData, trainSize);
    ResizeData(input, testData, testSize);
    trainLabel.set_size(trainSize);
    testLabel.set_size(testSize);

    using Col = arma::Col<size_t>;
    arma_rng::set_seed(seed);
    Col const sequence = arma::linspace<Col>(0, ExtractSize(input) - 1,
                                             ExtractSize(input));
    arma::Col<size_t> const order = arma::shuffle(sequence);

    for(size_t i = 0; i != trainSize; ++i)
    {
      ExtractData(input, trainData, order[i], i);
      trainLabel(i) = inputLabel(order[i]);
    }

    for(size_t i = 0; i != testSize; ++i)
    {
      ExtractData(input, testData,
                  order[i + trainSize], i);
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
  template<typename T>
  std::tuple<T, T,
  arma::Row<size_t>, arma::Row<size_t>>
  Split(T const &input,
        arma::Row<size_t> const &inputLabel)
  {
    T trainData;
    T testData;
    arma::Row<size_t> trainLabel;
    arma::Row<size_t> testLabel;

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
  template<typename T>
  void ExtractData(arma::Mat<T> const &input, arma::Mat<T> &output,
                   size_t inputIndex, size_t outputIndex) const
  {
    output.col(outputIndex) = input.col(inputIndex);
  }

  template<typename T>
  void ExtractData(arma::Cube<T> const &input, arma::Cube<T> &output,
                   size_t inputIndex, size_t outputIndex) const
  {
    outputIndex *= slice;
    inputIndex *= slice;
    output.slices(outputIndex, outputIndex + slice - 1) =
        input.slices(inputIndex, inputIndex + slice - 1);
  }

  template<typename T>
  size_t ExtractSize(arma::Mat<T> const &input) const
  {
    return input.n_cols;
  }

  template<typename T>
  size_t ExtractSize(arma::Cube<T> const &input) const
  {
    return input.n_slices;
  }

  template<typename T>
  void ResizeData(arma::Mat<T> const &input,
                  arma::Mat<T> &output,
                  size_t dataSize) const
  {
    output.set_size(input.n_rows, dataSize);
  }

  template<typename T>
  void ResizeData(arma::Cube<T> const &input,
                  arma::Cube<T> &output,
                  size_t dataSize) const
  {
    output.set_size(input.n_rows, input.n_cols, dataSize);
  }

  arma::arma_rng::seed_type seed;
  size_t slice;
  double testRatio;
};

} // namespace util
} // namespace mlpack

#endif
