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
   * @param blockHeight height of block
   * @param blockWidth width of block
   * @warning blockHeight * blockWidth must equal to maximalInputs.n_rows
   * By default the blockHeight and blockWidth will equal to
   * std::sqrt(maximalInputs.n_rows)
   * @code
   * arma::mat maximalInputs;
   * maximalInputs<<-1.0000<<0.1429<<arma::endr
   *              <<-0.7143<<0.4286<<arma::endr
   *              <<-0.4286<<0.7143<<arma::endr
   *              <<-0.1429<<1.0000<<arma::endr;
   * arma::mat output;
   * mlpack::math::ColumnsToBlocks ctb(1, 2);
   * ctb.Transform(maximalInputs, output);
   * //The cols of the maximalInputs output will reshape as a square which
   * //surrounded by padding value -1(this value could be set by BufValue)
   * //-1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000
   * //-1.0000  -1.0000  -0.4286  -1.0000   0.1429   0.7143  -1.0000
   * //-1.0000  -0.7143  -0.1429  -1.0000   0.4286   1.0000  -1.0000
   * //-1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000
   *
   * ctb.BlockWidth(4);
   * ctb.BlockHeight(1);
   * ctb.Transform(maxinalInputs, output);
   * //The cols of the maximalInputs output will reshape as a square which
   * //surrounded by padding value -1(this value could be set by BufValue)
   * //-1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000  -1.0000
   * //-1.0000 -1.0000 -0.7143 -0.4286 -0.1429 -1.0000  0.1429  0.4286  0.7143  1.0000  -1.0000
   * //-1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000  -1.0000
   *
   * @endcode
   */
  ColumnsToBlocks(size_t rows, size_t cols,
                  size_t blockHeight = 0,
                  size_t blockWidth = 0);

  /**
   * Transform the columns of maximalInputs into blocks
   * @param maximalInputs input value intent to transform to block, the rows
   * of this input must be a perfect square
   * @param output input transformed to block
   */
  void Transform(const arma::mat &maximalInputs, arma::mat &output);

  /**
   * Height of the block, please refer to the comments of
   * constructor for details
   */
  void BlockHeight(size_t value) {
    blockHeight = value;
  }

  /**
   * Return block height
   */
  size_t BlockHeight() const {
    return blockHeight;
  }

  /**
   * Width of the blcok, , please refer to the comments of
   * constructor for details
   */
  void BlockWidth(size_t value) {
    blockWidth = value;
  }

  /**
   * Return block width
   */
  size_t BlockWidth() const {
    return blockWidth;
  }

  /**
   * Get the size of the buffer, this size determine the how many cells surround
   * each column of the maximalInputs.
   * @param value buffer size, default value is 1
   */
  void BufSize(size_t value) {
    bufSize = value;
  }

  /**
   * Return buffer size
   */
  size_t BufSize() const {
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
  void Cols(size_t value) {
    cols = value;
  }

  /**
   * Return number of blocks per rows
   */
  size_t Cols() const {
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
  void Rows(size_t value) {
    rows = value;
  }

  /**
   * Return number of blocks per rows
   */
  size_t Rows() const {
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
  bool IsPerfectSquare(size_t value) const;

  size_t blockHeight;
  size_t blockWidth;
  size_t bufSize;
  double bufValue;
  size_t cols;
  double maxRange;
  double minRange;
  bool scale;
  size_t rows;
};

} // namespace math
} // namespace mlpack

#endif
