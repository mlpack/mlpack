
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add.hpp:

Program Listing for File add.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/add.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ADD_HPP
   #define MLPACK_METHODS_ANN_LAYER_ADD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Add
   {
    public:
     Add(const size_t outSize = 0);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     template<typename eT>
     void Gradient(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& error,
                   arma::Mat<eT>& gradient);
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     size_t OutputSize() const { return outSize; }
   
     size_t WeightSize() const { return outSize; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t outSize;
   
     OutputDataType weights;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   }; // class Add
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "add_impl.hpp"
   
   #endif
