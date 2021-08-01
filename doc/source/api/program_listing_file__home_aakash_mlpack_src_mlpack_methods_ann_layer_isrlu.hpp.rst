
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_isrlu.hpp:

Program Listing for File isrlu.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_isrlu.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/isrlu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author  = {Carlile, Brad and Delamarter, Guy and Kinney, Paul and Marti,
                Akiko and Whitney, Brian},
     title   = {Improving deep learning by inverse square root linear units (ISRLUs)},
     year    = {2017},
     url     = {https://arxiv.org/pdf/1710.09967.pdf}
   }
   
   #ifndef MLPACK_METHODS_ANN_LAYER_ISRLU_HPP
   #define MLPACK_METHODS_ANN_LAYER_ISRLU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class ISRLU
   {
    public:
     ISRLU(const double alpha = 1.0);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename DataType>
     void Backward(const DataType& input, const DataType& gy, DataType& g);
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     double const& Alpha() const { return alpha; }
     double& Alpha() { return alpha; }
   
     size_t WeightSize() { return 0; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     arma::mat derivative;
   
     double alpha;
   }; // class ISRLU
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "isrlu_impl.hpp"
   
   #endif
