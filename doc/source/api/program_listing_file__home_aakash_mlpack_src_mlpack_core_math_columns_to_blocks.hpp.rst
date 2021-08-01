
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_columns_to_blocks.hpp:

Program Listing for File columns_to_blocks.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_columns_to_blocks.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/columns_to_blocks.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NN_COLUMNS_TO_BLOCKS_HPP
   #define MLPACK_METHODS_NN_COLUMNS_TO_BLOCKS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace math {
   
   class ColumnsToBlocks
   {
    public:
     ColumnsToBlocks(size_t rows,
                     size_t cols,
                     size_t blockHeight = 0,
                     size_t blockWidth = 0);
   
     void Transform(const arma::mat& maximalInputs, arma::mat& output);
   
     void BlockHeight(const size_t value) { blockHeight = value; }
     size_t BlockHeight() const { return blockHeight; }
   
     void BlockWidth(size_t value) { blockWidth = value; }
     size_t BlockWidth() const { return blockWidth; }
   
     void BufSize(const size_t value) { bufSize = value; }
     size_t BufSize() const { return bufSize; }
   
     void BufValue(const double value) { bufValue = value; }
     double BufValue() const { return bufValue; }
   
     void MaxRange(const double value) { maxRange = value; }
     double MaxRange() const { return maxRange; }
   
     void MinRange(const double value) { minRange = value; }
     double MinRange() const { return minRange; }
   
     void Scale(const bool value) { scale = value; }
     bool Scale() const { return scale; }
   
     void Rows(const size_t value) { rows = value; }
     size_t Rows() const { return rows; }
   
     void Cols(const size_t value) { cols = value; }
     size_t Cols() const { return cols; }
   
    private:
     bool IsPerfectSquare(size_t value) const;
   
     size_t blockHeight;
     size_t blockWidth;
     size_t bufSize;
     double bufValue;
     double minRange;
     double maxRange;
     bool scale;
     size_t rows;
     size_t cols;
   };
   
   } // namespace math
   } // namespace mlpack
   
   #endif
