
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_max_pooling.hpp:

Program Listing for File max_pooling.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_max_pooling.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/max_pooling.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP
   #define MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   /*
    * The max pooling rule for convolution neural networks. Take the maximum value
    * within the receptive block.
    */
   class MaxPoolingRule
   {
    public:
     /*
      * Return the maximum value within the receptive block.
      *
      * @param input Input used to perform the pooling operation.
      */
     template<typename MatType>
     size_t Pooling(const MatType& input)
     {
       return arma::as_scalar(arma::find(input.max() == input, 1));
     }
   };
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class MaxPooling
   {
    public:
     MaxPooling();
   
     MaxPooling(const size_t kernelWidth,
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
   
     const OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     const OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t InputWidth() const { return inputWidth; }
     size_t& InputWidth() { return inputWidth; }
   
     size_t InputHeight() const { return inputHeight; }
     size_t& InputHeight() { return inputHeight; }
   
     size_t OutputWidth() const { return outputWidth; }
     size_t& OutputWidth() { return outputWidth; }
   
     size_t OutputHeight() const { return outputHeight; }
     size_t& OutputHeight() { return outputHeight; }
   
     size_t InputSize() const { return inSize; }
   
     size_t OutputSize() const { return outSize; }
   
     size_t KernelWidth() const { return kernelWidth; }
     size_t& KernelWidth() { return kernelWidth; }
   
     size_t KernelHeight() const { return kernelHeight; }
     size_t& KernelHeight() { return kernelHeight; }
   
     size_t StrideWidth() const { return strideWidth; }
     size_t& StrideWidth() { return strideWidth; }
   
     size_t StrideHeight() const { return strideHeight; }
     size_t& StrideHeight() { return strideHeight; }
   
     bool Floor() const { return floor; }
     bool& Floor() { return floor; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     size_t WeightSize() const { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     template<typename eT>
     void PoolingOperation(const arma::Mat<eT>& input,
                           arma::Mat<eT>& output,
                           arma::Mat<eT>& poolingIndices)
     {
       for (size_t j = 0, colidx = 0; j < output.n_cols;
           ++j, colidx += strideHeight)
       {
         for (size_t i = 0, rowidx = 0; i < output.n_rows;
             ++i, rowidx += strideWidth)
         {
           size_t rowEnd = rowidx + kernelWidth - 1;
           size_t colEnd = colidx + kernelHeight - 1;
   
           if (rowEnd > input.n_rows - 1)
             rowEnd = input.n_rows - 1;
           if (colEnd > input.n_cols - 1)
             colEnd = input.n_cols - 1;
   
           arma::mat subInput = input(
               arma::span(rowidx, rowEnd),
               arma::span(colidx, colEnd));
   
           const size_t idx = pooling.Pooling(subInput);
           output(i, j) = subInput(idx);
   
           if (!deterministic)
           {
             arma::Mat<size_t> subIndices = indices(arma::span(rowidx, rowEnd),
                 arma::span(colidx, colEnd));
   
             poolingIndices(i, j) = subIndices(idx);
           }
         }
       }
     }
   
     template<typename eT>
     void Unpooling(const arma::Mat<eT>& error,
                    arma::Mat<eT>& output,
                    arma::Mat<eT>& poolingIndices)
     {
       for (size_t i = 0; i < poolingIndices.n_elem; ++i)
       {
         output(poolingIndices(i)) += error(i);
       }
     }
   
     size_t kernelWidth;
   
     size_t kernelHeight;
   
     size_t strideWidth;
   
     size_t strideHeight;
   
     bool floor;
   
     size_t inSize;
   
     size_t outSize;
   
     bool reset;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     size_t outputWidth;
   
     size_t outputHeight;
   
     bool deterministic;
   
   
     size_t batchSize;
   
     arma::cube outputTemp;
   
     arma::cube inputTemp;
   
     arma::cube gTemp;
   
     MaxPoolingRule pooling;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   
     arma::Mat<size_t> indices;
   
     arma::Col<size_t> indicesCol;
   
     std::vector<arma::cube> poolingIndices;
   }; // class MaxPooling
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "max_pooling_impl.hpp"
   
   #endif
