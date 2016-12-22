/**
 * @file columns_to_blocks.hpp
 * @author Tham Ngap Wei
 *
 * A helper class that could be useful for visualizing the output of
 * MaximalInputs() and possibly other things.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NN_COLUMNS_TO_BLOCKS_HPP
#define MLPACK_METHODS_NN_COLUMNS_TO_BLOCKS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace math {

/**
 * Transform the columns of the given matrix into a block format.  This could be
 * useful with the mlpack::nn::MaximalInputs() function, if your training
 * samples are images.  Roughly speaking, given a matrix
 *
 * [[A]
 *  [B]
 *  [C]
 *  [D]]
 *
 * then the ColumnsToBlocks class can transform this to something like
 *
 * [[m m m m m]
 *  [m A m B m]
 *  [m m m m m]
 *  [m C m D m]
 *  [m m m m m]]
 *
 * where A through D are vectors and may themselves be reshaped by
 * ColumnsToBlocks.
 *
 * An example usage of the ColumnsToBlocks class with the output of
 * MaximalInputs() is given below; this assumes that the images are square, and
 * will return a matrix with a one-element margin, with each maximal input (that
 * is, each column of the maximalInput matrix) as a square block in the output
 * matrix.  5 rows and columns of blocks will be in the output matrix.
 *
 * @code
 * // We assume we have a sparse autoencoder 'encoder'.
 * arma::mat maximalInput; // Store the features learned by sparse autoencoder
 * mlpack::nn::MaximalInputs(encoder.Parameters(), maximalInput);
 *
 * arma::mat outputs;
 * const bool scale = true;
 *
 * ColumnsToBlocks ctb(5, 5);
 * arma::mat output;
 * ctb.Transform(maximalInput, output);
 * // You can save the output as a pgm, this may help you visualize the training
 * // results.
 * output.save(fileName, arma::pgm_binary);
 * @endcode
 *
 * Another example of usage is given below, on a sample matrix.
 *
 * @code
 * // This matrix has two columns.
 * arma::mat input;
 * input << -1.0000 << 0.1429 << arma::endr
 *       << -0.7143 << 0.4286 << arma::endr
 *       << -0.4286 << 0.7143 << arma::endr
 *       << -0.1429 << 1.0000 << arma::endr;
 *
 * arma::mat output;
 * ColumnsToBlocks ctb(1, 2);
 * ctb.Transform(input, output);
 *
 * // The columns of the input will be reshaped as a square which is
 * // surrounded by padding value -1 (this value could be changed with the
 * // BufValue() method):
 * // -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000
 * // -1.0000  -1.0000  -0.4286  -1.0000   0.1429   0.7143  -1.0000
 * // -1.0000  -0.7143  -0.1429  -1.0000   0.4286   1.0000  -1.0000
 * // -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000
 *
 * // Now, let's change some parameters; let's have each input column output not
 * // as a square, but as a 4x1 vector.
 * ctb.BlockWidth(1);
 * ctb.BlockHeight(4);
 * ctb.Transform(input, output);
 *
 * // The output here will be similar, but each maximal input is 4x1:
 * // -1.0000 -1.0000 -1.0000 -1.0000 -1.0000
 * // -1.0000 -1.0000 -1.0000  0.1429 -1.0000
 * // -1.0000 -0.7143 -1.0000  0.4286 -1.0000
 * // -1.0000 -0.4286 -1.0000  0.7143 -1.0000
 * // -1.0000 -0.1429 -1.0000  1.0000 -1.0000
 * // -1.0000 -1.0000 -1.0000 -1.0000 -1.0000
 * @endcode
 *
 * The ColumnsToBlocks class can also, depending on the parameters, scale the
 * input to a given range (useful for exporting to PGM, for instance), and also
 * set the buffer size and value.  See the Scale(), MinRange(), MaxRange(),
 * BufSize(), and BufValue() methods for more details.
 */
class ColumnsToBlocks
{
 public:
  /**
   * Constructor a ColumnsToBlocks object with the given parameters.  The rows
   * and cols parameters control the number of blocks per row and column of the
   * output matrix, respectively, and the blockHeight and blockWidth parameters
   * control the size of the individual blocks.  If blockHeight and blockWidth
   * are specified, then (blockHeight * blockWidth) must be equal to the number
   * of rows in the input matrix when Transform() is called.  If blockHeight and
   * blockWidth are not specified, then the square root of the number of rows of
   * the input matrix will be taken when Transform() is called and that will be
   * used as the block width and height.
   *
   * Note that the ColumnsToBlocks object can also scale the inputs to a given
   * range; see Scale(), MinRange(), and MaxRange(), and the buffer (margin)
   * size can also be set with BufSize(), and the value used for the buffer can
   * be set with BufValue().
   *
   * @param rows Number of blocks in each column of the output matrix.
   * @param cols Number of blocks in each row of the output matrix.
   * @param blockHeight Height of each block.
   * @param blockWidth Width of each block.
   *
   * @warning blockHeight * blockWidth must be equal to maximalInputs.n_rows.
   */
  ColumnsToBlocks(size_t rows,
                  size_t cols,
                  size_t blockHeight = 0,
                  size_t blockWidth = 0);

  /**
   * Transform the columns of the input matrix into blocks.  If blockHeight and
   * blockWidth were not specified in the constructor (and BlockHeight() and
   * BlockWidth() were not called), then the number of rows in the input matrix
   * must be a perfect square.
   *
   * @param input Input matrix to transform.
   * @param output Matrix to store transformed output in.
   */
  void Transform(const arma::mat& maximalInputs, arma::mat& output);

  //! Set the height of each block; see the constructor for more details.
  void BlockHeight(const size_t value) { blockHeight = value; }
  //! Get the block height.
  size_t BlockHeight() const { return blockHeight; }

  //! Set the width of each block; see the constructor for more details.
  void BlockWidth(size_t value) { blockWidth = value; }
  //! Get the block width.
  size_t BlockWidth() const { return blockWidth; }

  //! Modify the buffer size (the size of the margin around each column of the
  //! input).  The default value is 1.
  void BufSize(const size_t value) { bufSize = value; }
  //! Get the buffer size.
  size_t BufSize() const { return bufSize; }

  //! Modify the value used for buffer cells; the default is -1.
  void BufValue(const double value) { bufValue = value; }
  //! Get the value used for buffer cells.
  double BufValue() const { return bufValue; }

  //! Set the maximum of the range the input will be scaled to, if scaling is
  //! enabled (see Scale()).
  void MaxRange(const double value) { maxRange = value; }
  //! Get the maximum of the range the input will be scaled to, if scaling is
  //! enabled (see Scale()).
  double MaxRange() const { return maxRange; }

  //! Set the minimum of the range the input will be scaled to, if scaling is
  //! enabled (see Scale()).
  void MinRange(const double value) { minRange = value; }
  //! Get the minimum of the range the input will be scaled to, if scaling is
  //! enabled (see Scale()).
  double MinRange() const { return minRange; }

  //! Set whether or not scaling is enabled (see also MaxRange() and
  //! MinRange()).
  void Scale(const bool value) { scale = value; }
  //! Get whether or not scaling is enabled (see also MaxRange() and
  //! MinRange()).
  bool Scale() const { return scale; }

  //! Set the number of blocks per row.
  void Rows(const size_t value) { rows = value; }
  //! Modify the number of blocks per row.
  size_t Rows() const { return rows; }

  //! Set the number of blocks per column.
  void Cols(const size_t value) { cols = value; }
  //! Return the number of blocks per column.
  size_t Cols() const { return cols; }

 private:
  //! Determine whether or not the number is a perfect square.
  bool IsPerfectSquare(size_t value) const;

  //! The height of each block.
  size_t blockHeight;
  //! The width of each block.
  size_t blockWidth;
  //! The size of the buffer around each block.
  size_t bufSize;
  //! The value of the buffer around each block.
  double bufValue;
  //! The minimum of the range to be scaled to (if scaling is enabled).
  double minRange;
  //! The maximum of the range to be scaled to (if scaling is enabled).
  double maxRange;
  //! Whether or not scaling is enabled.
  bool scale;
  //! The number of blocks in each row.
  size_t rows;
  //! The number of blocks in each column.
  size_t cols;
};

} // namespace math
} // namespace mlpack

#endif
