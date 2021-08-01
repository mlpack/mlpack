
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_select.hpp:

Program Listing for File select.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_select.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/select.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SELECT_HPP
   #define MLPACK_METHODS_ANN_LAYER_SELECT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Select
   {
    public:
     Select(const size_t index = 0, const size_t elements = 0);
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t const& Index() const { return index; }
   
     size_t const& NumElements() const { return elements; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t index;
   
     size_t elements;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class Select
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "select_impl.hpp"
   
   #endif
