
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_atrous_convolution.hpp:

Program Listing for File atrous_convolution.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_atrous_convolution.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/atrous_convolution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_HPP
   #define MLPACK_METHODS_ANN_LAYER_ATROUS_CONVOLUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
   #include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>
   #include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>
   #include <mlpack/methods/ann/convolution_rules/svd_convolution.hpp>
   #include <mlpack/core/util/to_lower.hpp>
   
   #include "layer_types.hpp"
   #include "padding.hpp"
   
   namespace mlpack{
   namespace ann  {
   
   template <
       typename ForwardConvolutionRule = NaiveConvolution<ValidConvolution>,
       typename BackwardConvolutionRule = NaiveConvolution<FullConvolution>,
       typename GradientConvolutionRule = NaiveConvolution<ValidConvolution>,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class AtrousConvolution
   {
    public:
     AtrousConvolution();
   
     AtrousConvolution(const size_t inSize,
                       const size_t outSize,
                       const size_t kernelWidth,
                       const size_t kernelHeight,
                       const size_t strideWidth = 1,
                       const size_t strideHeight = 1,
                       const size_t padW = 0,
                       const size_t padH = 0,
                       const size_t inputWidth = 0,
                       const size_t inputHeight = 0,
                       const size_t dilationWidth = 1,
                       const size_t dilationHeight = 1,
                       const std::string& paddingType = "None");
   
     AtrousConvolution(const size_t inSize,
                       const size_t outSize,
                       const size_t kernelWidth,
                       const size_t kernelHeight,
                       const size_t strideWidth,
                       const size_t strideHeight,
                       const std::tuple<size_t, size_t>& padW,
                       const std::tuple<size_t, size_t>& padH,
                       const size_t inputWidth = 0,
                       const size_t inputHeight = 0,
                       const size_t dilationWidth = 1,
                       const size_t dilationHeight = 1,
                       const std::string& paddingType = "None");
   
     /*
      * Set the weight and bias term.
      */
     void Reset();
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     /*
      * Calculate the gradient using the output delta and the input activation.
      *
      * @param input The input parameter used for calculating the gradient.
      * @param error The calculated error.
      * @param gradient The calculated gradient.
      */
     template<typename eT>
     void Gradient(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     arma::cube const& Weight() const { return weight; }
     arma::cube& Weight() { return weight; }
   
     arma::mat const& Bias() const { return bias; }
     arma::mat& Bias() { return bias; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
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
   
     size_t DilationWidth() const { return dilationWidth; }
     size_t& DilationWidth() { return dilationWidth; }
   
     size_t DilationHeight() const { return dilationHeight; }
     size_t& DilationHeight() { return dilationHeight; }
   
     ann::Padding<> const& Padding() const { return padding; }
     ann::Padding<>& Padding() { return padding; }
   
     size_t WeightSize() const
     {
       return (outSize * inSize * kernelWidth * kernelHeight) + outSize;
     }
   
     size_t InputShape() const
     {
       return inputHeight * inputWidth * inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     /*
      * Return the convolution output size.
      *
      * @param size The size of the input (row or column).
      * @param k The size of the filter (width or height).
      * @param s The stride size (x or y direction).
      * @param pSideOne The size of the padding (width or height) on one side.
      * @param pSideTwo The size of the padding (width or height) on another side.
      * @param d The dilation size.
      * @return The convolution output size.
      */
     size_t ConvOutSize(const size_t size,
                        const size_t k,
                        const size_t s,
                        const size_t pSideOne,
                        const size_t pSideTwo,
                        const size_t d)
     {
       return std::floor(size + pSideOne + pSideTwo - d * (k - 1) - 1) / s + 1;
     }
   
     /*
      * Function to assign padding such that output size is same as input size.
      */
     void InitializeSamePadding(size_t& padWLeft,
                                size_t& padWRight,
                                size_t& padHBottom,
                                size_t& padHTop) const;
   
     /*
      * Rotates a 3rd-order tensor counterclockwise by 180 degrees.
      *
      * @param input The input data to be rotated.
      * @param output The rotated output.
      */
     template<typename eT>
     void Rotate180(const arma::Cube<eT>& input, arma::Cube<eT>& output)
     {
       output = arma::Cube<eT>(input.n_rows, input.n_cols, input.n_slices);
   
       // * left-right flip, up-down flip */
       for (size_t s = 0; s < output.n_slices; s++)
         output.slice(s) = arma::fliplr(arma::flipud(input.slice(s)));
     }
   
     /*
      * Rotates a dense matrix counterclockwise by 180 degrees.
      *
      * @param input The input data to be rotated.
      * @param output The rotated output.
      */
     template<typename eT>
     void Rotate180(const arma::Mat<eT>& input, arma::Mat<eT>& output)
     {
       // * left-right flip, up-down flip */
       output = arma::fliplr(arma::flipud(input));
     }
   
     size_t inSize;
   
     size_t outSize;
   
     size_t batchSize;
   
     size_t kernelWidth;
   
     size_t kernelHeight;
   
     size_t strideWidth;
   
     size_t strideHeight;
   
     OutputDataType weights;
   
     arma::cube weight;
   
     arma::mat bias;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     size_t outputWidth;
   
     size_t outputHeight;
   
     size_t dilationWidth;
   
     size_t dilationHeight;
   
     arma::cube outputTemp;
   
     arma::cube inputPaddedTemp;
   
     arma::cube gTemp;
   
     arma::cube gradientTemp;
   
     ann::Padding<> padding;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   }; // class AtrousConvolution
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "atrous_convolution_impl.hpp"
   
   #endif
