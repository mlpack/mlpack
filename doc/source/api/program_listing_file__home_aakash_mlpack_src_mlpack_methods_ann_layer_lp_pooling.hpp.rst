
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lp_pooling.hpp:

Program Listing for File lp_pooling.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lp_pooling.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/lp_pooling.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LP_POOLING_HPP
   #define MLPACK_METHODS_ANN_LAYER_LP_POOLING_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class LpPooling
   {
    public:
     LpPooling();
   
     LpPooling(const size_t normType,
               const size_t kernelWidth,
               const size_t kernelHeight,
               const size_t strideWidth = 1,
               const size_t strideHeight = 1,
               const bool floor = true);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t const& InputWidth() const { return inputWidth; }
     size_t& InputWidth() { return inputWidth; }
   
     size_t const& InputHeight() const { return inputHeight; }
     size_t& InputHeight() { return inputHeight; }
   
     size_t const& OutputWidth() const { return outputWidth; }
     size_t& OutputWidth() { return outputWidth; }
   
     size_t const& OutputHeight() const { return outputHeight; }
     size_t& OutputHeight() { return outputHeight; }
   
     size_t InputSize() const { return inSize; }
   
     size_t OutputSize() const { return outSize; }
   
     size_t NormType() const { return normType; }
     size_t& NormType() { return normType; }
   
     size_t KernelWidth() const { return kernelWidth; }
     size_t& KernelWidth() { return kernelWidth; }
   
     size_t KernelHeight() const { return kernelHeight; }
     size_t& KernelHeight() { return kernelHeight; }
   
     size_t StrideWidth() const { return strideWidth; }
     size_t& StrideWidth() { return strideWidth; }
   
     size_t StrideHeight() const { return strideHeight; }
     size_t& StrideHeight() { return strideHeight; }
   
     bool const& Floor() const { return floor; }
     bool& Floor() { return floor; }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     template<typename eT>
     void Pooling(const arma::Mat<eT>& input, arma::Mat<eT>& output)
     {
       arma::Mat<eT> inputPre = input;
       inputPre = arma::pow(inputPre, normType);
   
       for (size_t i = 1; i < input.n_cols; ++i)
         inputPre.col(i) += inputPre.col(i - 1);
   
       for (size_t i = 1; i < input.n_rows; ++i)
         inputPre.row(i) += inputPre.row(i - 1);
   
       for (size_t j = 0, colidx = 0; j < output.n_cols;
            ++j, colidx += strideHeight)
       {
         for (size_t i = 0, rowidx = 0; i < output.n_rows;
              ++i, rowidx += strideWidth)
         {
           double val = 0.0;
           size_t rowEnd = rowidx + kernelWidth - 1;
           size_t colEnd = colidx + kernelHeight - 1;
   
           if (rowEnd > input.n_rows - 1)
             rowEnd = input.n_rows - 1;
           if (colEnd > input.n_cols - 1)
             colEnd = input.n_cols - 1;
   
           val += inputPre(rowEnd, colEnd);
           if (rowidx >= 1)
           {
             if (colidx >= 1)
               val += inputPre(rowidx - 1, colidx - 1);
             val -= inputPre(rowidx - 1, colEnd);
           }
   
           if (colidx >= 1)
             val -= inputPre(rowEnd, colidx - 1);
   
           output(i, j) = val;
         }
       }
   
       output = arma::pow(output, 1.0 / normType);
     }
   
     template<typename eT>
     void Unpooling(const arma::Mat<eT>& input,
                    const arma::Mat<eT>& error,
                    arma::Mat<eT>& output)
     {
       arma::Mat<eT> unpooledError;
       for (size_t j = 0, colidx = 0; j < input.n_cols; j += strideHeight,
            colidx++)
       {
         for (size_t i = 0, rowidx = 0; i < input.n_rows; i += strideWidth,
              rowidx++)
         {
           size_t rowEnd = i + kernelWidth - 1;
           size_t colEnd = j + kernelHeight - 1;
   
           if (rowEnd > input.n_rows - 1)
           {
             if (floor)
               continue;
             rowEnd = input.n_rows - 1;
           }
   
           if (colEnd > input.n_cols - 1)
           {
             if (floor)
               continue;
             colEnd = input.n_cols - 1;
           }
   
           arma::mat InputArea = input(arma::span(i, rowEnd),
               arma::span(j, colEnd));
   
           size_t sum = pow(arma::accu(arma::pow(InputArea, normType)),
               (normType - 1) / normType);
           unpooledError = arma::Mat<eT>(InputArea.n_rows, InputArea.n_cols);
           unpooledError.fill(error(rowidx, colidx) / InputArea.n_elem);
           unpooledError %= arma::pow(InputArea, normType - 1);
           unpooledError /= sum;
           output(arma::span(i, i + InputArea.n_rows - 1),
               arma::span(j, j + InputArea.n_cols - 1)) += unpooledError;
         }
       }
     }
   
     size_t normType;
   
     size_t kernelWidth;
   
     size_t kernelHeight;
   
     size_t strideWidth;
   
     size_t strideHeight;
   
     bool floor;
   
     size_t inSize;
   
     size_t outSize;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     size_t outputWidth;
   
     size_t outputHeight;
   
     bool reset;
   
     size_t batchSize;
   
     arma::cube outputTemp;
   
     arma::cube inputTemp;
   
     arma::cube gTemp;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   }; // class LpPooling
   
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "lp_pooling_impl.hpp"
   
   #endif
