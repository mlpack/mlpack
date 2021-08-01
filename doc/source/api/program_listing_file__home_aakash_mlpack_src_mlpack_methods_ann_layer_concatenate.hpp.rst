
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concatenate.hpp:

Program Listing for File concatenate.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concatenate.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/concatenate.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CONCATENATE_HPP
   #define MLPACK_METHODS_ANN_LAYER_CONCATENATE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Concatenate
   {
    public:
     Concatenate();
   
     Concatenate(const Concatenate& layer);
   
     Concatenate(Concatenate&& layer);
   
     Concatenate& operator=(const Concatenate& layer);
   
     Concatenate& operator=(Concatenate&& layer);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Concat() const { return concat; }
     OutputDataType& Concat() { return concat; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */)
     {
       // Nothing to do here.
     }
   
    private:
     size_t inRows;
   
     OutputDataType weights;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     OutputDataType concat;
   }; // class Concatenate
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "concatenate_impl.hpp"
   
   #endif
