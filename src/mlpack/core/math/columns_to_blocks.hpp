#ifndef __MLPACK_METHODS_NN_COLUMNS_TO_BLOCKS_HPP
#define __MLPACK_METHODS_NN_COLUMNS_TO_BLOCKS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace math {

/**
 * Transform the output of "MaximalInputs" to blocks, if your training samples are images,
 * this function could help you visualize your training results
 * @param maximalInputs Parameters after maximize by "MaximalInputs", each col assiociate to one sample
 * @param output Maximal inputs regrouped to blocks
 * @param scale False, the output will not be scaled and vice versa
 * @param minRange minimum range of the output
 * @param maxRange maximum range of the output
 * @code
 * arma::mat maximalInput; //store the features learned by sparse autoencoder
 * mlpack::nn::MaximalInputs(encoder2.Parameters(), maximalInput);
 *
 * arma::mat outputs;
 * const bool scale = true;
 *
 * ColumnsToBlocks ctb(5,5);
 * arma::mat output;
 * ctb.Transform(maximalInput, output);
 * //you can save the output as a pgm, this may help you visualize the training results
 * output.save(fileName, arma::pgm_binary);
 * @endcode
 */
class ColumnsToBlocks
{
 public:
  /**
   * Constructor of ColumnsToBlocks
   * @param rows number of blocks per cols
   * @param cols number of blocks per rows
   */
  ColumnsToBlocks(arma::uword rows, arma::uword cols);

  /**
   * Transform the columns of maximalInputs into blocks
   * @param maximalInputs input value intent to transform to block, the rows
   * of this input must be a perfect square
   * @param output input transformed to block
   */
  void Transform(const arma::mat &maximalInputs, arma::mat &output);

  /**
   * Get the size of the buffer, this size determine the how many cells surround
   * each column of the maximalInputs.
   * @param value buffer size, default value is 1
   */
  void BufSize(arma::uword value) {
    bufSize = value;
  }

  /**
   * Return buffer size
   */
  arma::uword BufSize() const {
    return bufSize;
  }

  /**
   * Set the buffer value surround the cells
   * @param value The value of the buffer, default value is -1
   */
  void BufValue(double value) {
    bufValue = value;
  }
  double BufValue() const {
    return bufValue;
  }

  /**
   * set number of blocks per rows
   * @param value number of blocks per rows, default value is 0
   */
  void Cols(arma::uword value) {
    cols = value;
  }

  /**
   * Return number of blocks per rows
   */
  arma::uword Cols() const {
    return cols;
  }

  /**
   * Set maximum range for scaling
   * @param value maximum range, default value is 255
   */
  void MaxRange(double value) {
    maxRange = value;
  }

  /**
   * Return maximum range
   */
  double MaxRange() const {
    return maxRange;
  }

  /**
   * Set minimum range for scaling
   * @param value minimum range, default value is 0
   */
  void MinRange(double value) {
    minRange = value;
  }

  /**
   * Return minimum range
   */
  double MinRange() const {
    return minRange;
  }

  /**
   * @brief Set number of blocks per rows
   * @param cols number of blocks per rows, default value is 0
   */
  void Rows(arma::uword value) {
    rows = value;
  }

  /**
   * Return number of blocks per rows
   */
  arma::uword Rows() const {
    return rows;
  }

  /**
   * Disable or enable scale
   * @param value True, scale the output range and vice versa.Default
   * value is false
   */
  void Scale(bool value) {
    scale = value;
  }

  /**
   * Return scale value
   */
  bool Scale() const {
    return scale;
  }

 private:
  bool IsPerfectSquare(arma::uword value) const;

  arma::uword bufSize;
  double bufValue;
  arma::uword cols;
  double maxRange;
  double minRange;
  bool scale;
  arma::uword rows;
};

} // namespace math
} // namespace mlpack

#endif
