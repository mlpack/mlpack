
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_glimpse.hpp:

Program Listing for File glimpse.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_glimpse.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/glimpse.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author  = {Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu},
     title   = {Recurrent Models of Visual Attention},
     journal = {CoRR},
     volume  = {abs/1406.6247},
     year    = {2014},
     url     = {https://arxiv.org/abs/1406.6247}
   }
   
   #ifndef MLPACK_METHODS_ANN_LAYER_GLIMPSE_HPP
   #define MLPACK_METHODS_ANN_LAYER_GLIMPSE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "layer_types.hpp"
   #include <algorithm>
   
   namespace mlpack {
   namespace ann  {
   
   
   /*
    * The mean pooling rule for convolution neural networks. Average all values
    * within the receptive block.
    */
   class MeanPoolingRule
   {
    public:
     /*
      * Return the average value within the receptive block.
      *
      * @param input Input used to perform the pooling operation.
      */
     template<typename MatType>
     double Pooling(const MatType& input)
     {
       return arma::mean(arma::mean(input));
     }
   
     /*
      * Set the average value within the receptive block.
      *
      * @param input Input used to perform the pooling operation.
      * @param value The unpooled value.
      * @param output The unpooled output data.
      */
     template<typename MatType>
     void Unpooling(const MatType& input, const double value, MatType& output)
     {
       output = arma::zeros<MatType>(input.n_rows, input.n_cols);
       const double mean = arma::mean(arma::mean(input));
   
       output.elem(arma::find(mean == input, 1)).fill(value);
     }
   };
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Glimpse
   {
    public:
     Glimpse(const size_t inSize = 0,
             const size_t size = 0,
             const size_t depth = 3,
             const size_t scale = 2,
             const size_t inputWidth = 0,
             const size_t inputHeight = 0);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType& OutputParameter() const {return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     void Location(const arma::mat& location)
     {
       this->location = location;
     }
   
     size_t const& InputWidth() const { return inputWidth; }
     size_t& InputWidth() { return inputWidth; }
   
     size_t const& InputHeight() const { return inputHeight; }
     size_t& InputHeight() { return inputHeight; }
   
     size_t const& OutputWidth() const { return outputWidth; }
     size_t& OutputWidth() { return outputWidth; }
   
     size_t const& OutputHeight() const { return outputHeight; }
     size_t& OutputHeight() { return outputHeight; }
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     size_t const& Depth() const { return depth; }
   
     size_t const& Scale() const { return scale; }
   
     size_t InSize() const { return inSize; }
   
     size_t GlimpseSize() const { return size;}
   
     size_t InputShape() const
     {
       return inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     /*
      * Transform the given input by changing rows to columns.
      *
      * @param w The input matrix used to perform the transformation.
      */
     void Transform(arma::mat& w)
     {
       arma::mat t = w;
   
       for (size_t i = 0, k = 0; i < w.n_elem; ++k)
       {
         for (size_t j = 0; j < w.n_cols; ++j, ++i)
         {
           w(k, j) = t(i);
         }
       }
     }
   
     /*
      * Transform the given input by changing rows to columns.
      *
      * @param w The input matrix used to perform the transformation.
      */
     void Transform(arma::cube& w)
     {
       for (size_t i = 0; i < w.n_slices; ++i)
       {
         arma::mat t = w.slice(i);
         Transform(t);
         w.slice(i) = t;
       }
     }
   
     template<typename eT>
     void Pooling(const size_t kSize,
                  const arma::Mat<eT>& input,
                  arma::Mat<eT>& output)
     {
       const size_t rStep = kSize;
       const size_t cStep = kSize;
   
       for (size_t j = 0; j < input.n_cols; j += cStep)
       {
         for (size_t i = 0; i < input.n_rows; i += rStep)
         {
           output(i / rStep, j / cStep) += pooling.Pooling(
               input(arma::span(i, i + rStep - 1), arma::span(j, j + cStep - 1)));
         }
       }
     }
   
     template<typename eT>
     void Unpooling(const arma::Mat<eT>& input,
                    const arma::Mat<eT>& error,
                    arma::Mat<eT>& output)
     {
       const size_t rStep = input.n_rows / error.n_rows;
       const size_t cStep = input.n_cols / error.n_cols;
   
       arma::Mat<eT> unpooledError;
       for (size_t j = 0; j < input.n_cols; j += cStep)
       {
         for (size_t i = 0; i < input.n_rows; i += rStep)
         {
           const arma::Mat<eT>& inputArea = input(arma::span(i, i + rStep - 1),
                                                  arma::span(j, j + cStep - 1));
   
           pooling.Unpooling(inputArea, error(i / rStep, j / cStep),
               unpooledError);
   
           output(arma::span(i, i + rStep - 1),
               arma::span(j, j + cStep - 1)) += unpooledError;
         }
       }
     }
   
     template<typename eT>
     void ReSampling(const arma::Mat<eT>& input, arma::Mat<eT>& output)
     {
       double wRatio = (double) (input.n_rows - 1) / (size - 1);
       double hRatio = (double) (input.n_cols - 1) / (size - 1);
   
       double iWidth = input.n_rows - 1;
       double iHeight = input.n_cols - 1;
   
       for (size_t y = 0; y < size; y++)
       {
         for (size_t x = 0; x < size; x++)
         {
           double ix = wRatio * x;
           double iy = hRatio * y;
   
           // Get the 4 nearest neighbors.
           double ixNw = std::floor(ix);
           double iyNw = std::floor(iy);
           double ixNe = ixNw + 1;
           double iySw = iyNw + 1;
   
           // Get surfaces to each neighbor.
           double se = (ix - ixNw) * (iy - iyNw);
           double sw = (ixNe - ix) * (iy - iyNw);
           double ne = (ix - ixNw) * (iySw - iy);
           double nw = (ixNe - ix) * (iySw - iy);
   
           // Calculate the weighted sum.
           output(y, x) = input(iyNw, ixNw) * nw +
               input(iyNw, std::min(ixNe,  iWidth)) * ne +
               input(std::min(iySw, iHeight), ixNw) * sw +
               input(std::min(iySw, iHeight), std::min(ixNe, iWidth)) * se;
         }
       }
     }
   
     template<typename eT>
     void DownwardReSampling(const arma::Mat<eT>& input,
                             const arma::Mat<eT>& error,
                             arma::Mat<eT>& output)
     {
       double iWidth = input.n_rows - 1;
       double iHeight = input.n_cols - 1;
   
       double wRatio = iWidth / (size - 1);
       double hRatio = iHeight / (size - 1);
   
       for (size_t y = 0; y < size; y++)
       {
         for (size_t x = 0; x < size; x++)
         {
           double ix = wRatio * x;
           double iy = hRatio * y;
   
           // Get the 4 nearest neighbors.
           double ixNw = std::floor(ix);
           double iyNw = std::floor(iy);
           double ixNe = ixNw + 1;
           double iySw = iyNw + 1;
   
           // Get surfaces to each neighbor.
           double se = (ix - ixNw) * (iy - iyNw);
           double sw = (ixNe - ix) * (iy - iyNw);
           double ne = (ix - ixNw) * (iySw - iy);
           double nw = (ixNe - ix) * (iySw - iy);
   
           double ograd = error(y, x);
   
           output(iyNw, ixNw) = output(iyNw, ixNw) + nw * ograd;
           output(iyNw, std::min(ixNe, iWidth)) = output(iyNw,
               std::min(ixNe, iWidth)) + ne * ograd;
           output(std::min(iySw, iHeight), ixNw) = output(std::min(iySw, iHeight),
               ixNw) + sw * ograd;
           output(std::min(iySw, iHeight), std::min(ixNe, iWidth)) = output(
               std::min(iySw, iHeight), std::min(ixNe, iWidth)) + se * ograd;
         }
       }
     }
   
     size_t inSize;
   
     size_t size;
   
     size_t depth;
   
     size_t scale;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     size_t outputWidth;
   
     size_t outputHeight;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     size_t inputDepth;
   
     arma::cube inputTemp;
   
     arma::cube outputTemp;
   
     arma::mat location;
   
     MeanPoolingRule pooling;
   
     std::vector<arma::mat> locationParameter;
   
     arma::cube gTemp;
   
     bool deterministic;
   }; // class GlimpseLayer
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "glimpse_impl.hpp"
   
   #endif
