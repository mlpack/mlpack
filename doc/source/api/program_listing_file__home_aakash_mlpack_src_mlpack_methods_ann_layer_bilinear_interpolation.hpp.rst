
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_bilinear_interpolation.hpp:

Program Listing for File bilinear_interpolation.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_bilinear_interpolation.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/bilinear_interpolation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_BILINEAR_INTERPOLATION_HPP
   #define MLPACK_METHODS_ANN_LAYER_BILINEAR_INTERPOLATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class BilinearInterpolation
   {
    public:
     BilinearInterpolation();
   
     BilinearInterpolation(const size_t inRowSize,
                           const size_t inColSize,
                           const size_t outRowSize,
                           const size_t outColSize,
                           const size_t depth);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /*input*/,
                   const arma::Mat<eT>& gradient,
                   arma::Mat<eT>& output);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t const& InRowSize() const { return inRowSize; }
     size_t& InRowSize() { return inRowSize; }
   
     size_t const& InColSize() const { return inColSize; }
     size_t& InColSize() { return inColSize; }
   
     size_t const& OutRowSize() const { return outRowSize; }
     size_t& OutRowSize() { return outRowSize; }
   
     size_t const& OutColSize() const { return outColSize; }
     size_t& OutColSize() { return outColSize; }
   
     size_t const& InDepth() const { return depth; }
     size_t& InDepth() { return depth; }
   
     size_t InputShape() const
     {
       return inRowSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inRowSize;
     size_t inColSize;
     size_t outRowSize;
     size_t outColSize;
     size_t depth;
     size_t batchSize;
     OutputDataType delta;
     OutputDataType outputParameter;
   }; // class BilinearInterpolation
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "bilinear_interpolation_impl.hpp"
   
   #endif
