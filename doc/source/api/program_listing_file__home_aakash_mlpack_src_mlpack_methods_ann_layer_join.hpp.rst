
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_join.hpp:

Program Listing for File join.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_join.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/join.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_JOIN_HPP
   #define MLPACK_METHODS_ANN_LAYER_JOIN_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template<
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Join
   {
    public:
     Join();
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inSizeRows;
   
     size_t inSizeCols;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class Join
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "join_impl.hpp"
   
   #endif
