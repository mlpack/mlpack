
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_padding.hpp:

Program Listing for File padding.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_padding.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/padding.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_PADDING_HPP
   #define MLPACK_METHODS_ANN_LAYER_PADDING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Padding
   {
    public:
     Padding(const size_t padWLeft = 0,
             const size_t padWRight = 0,
             const size_t padHTop = 0,
             const size_t padHBottom = 0,
             const size_t inputWidth = 0,
             const size_t inputHeight = 0);
   
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
   
     size_t PadWLeft() const { return padWLeft; }
     size_t& PadWLeft() { return padWLeft; }
   
     size_t PadWRight() const { return padWRight; }
     size_t& PadWRight() { return padWRight; }
   
     size_t PadHTop() const { return padHTop; }
     size_t& PadHTop() { return padHTop; }
   
     size_t PadHBottom() const { return padHBottom; }
     size_t& PadHBottom() { return padHBottom; }
   
     size_t InputWidth() const { return inputWidth; }
     size_t& InputWidth() { return inputWidth; }
   
     size_t InputHeight() const { return inputHeight; }
     size_t& InputHeight() { return inputHeight; }
   
     size_t OutputWidth() const { return outputWidth; }
     size_t& OutputWidth() { return outputWidth; }
   
     size_t OutputHeight() const { return outputHeight; }
     size_t& OutputHeight() { return outputHeight; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t padWLeft;
   
     size_t padWRight;
   
     size_t padHTop;
   
     size_t padHBottom;
   
     size_t nRows, nCols;
   
     size_t inputHeight;
   
     size_t inputWidth;
     
     size_t outputHeight;
     
     size_t outputWidth;
   
     size_t inSize;
   
     arma::cube inputTemp;
   
     arma::cube outputTemp;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class Padding
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "padding_impl.hpp"
   
   #endif
