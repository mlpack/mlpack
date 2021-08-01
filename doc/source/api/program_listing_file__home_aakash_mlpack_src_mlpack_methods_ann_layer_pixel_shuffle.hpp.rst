
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_pixel_shuffle.hpp:

Program Listing for File pixel_shuffle.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_pixel_shuffle.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/pixel_shuffle.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_HPP
   #define MLPACK_METHODS_ANN_LAYER_PIXEL_SHUFFLE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class PixelShuffle
   {
    public:
     PixelShuffle();
   
     PixelShuffle(const size_t upscaleFactor,
                  const size_t height,
                  const size_t width,
                  const size_t size);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t UpscaleFactor() const { return upscaleFactor; }
   
     size_t& UpscaleFactor() { return upscaleFactor; }
   
     size_t InputHeight() const { return height; }
   
     size_t& InputHeight() { return height; }
   
     size_t InputWidth() const { return width; }
   
     size_t& InputWidth() { return width; }
   
     size_t InputChannels() const { return size; }
   
     size_t& InputChannels() { return size; }
   
     size_t OutputHeight() const { return outputHeight; }
   
     size_t OutputWidth() const { return outputWidth; }
   
     size_t OutputChannels() const { return sizeOut; }
   
     template<typename Archive>
     void serialize(Archive& ar, const unsigned int /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     size_t upscaleFactor;
   
     size_t height;
   
     size_t width;
   
     size_t size;
   
     size_t batchSize;
   
     size_t outputHeight;
   
     size_t outputWidth;
   
     size_t sizeOut;
   
     bool reset;
   }; // class PixelShuffle
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "pixel_shuffle_impl.hpp"
   
   #endif
