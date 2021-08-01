
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout.hpp:

Program Listing for File alpha_dropout.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/alpha_dropout.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_HPP
   #define MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <typename InputDataType = arma::mat,
             typename OutputDataType = arma::mat>
   class AlphaDropout
   {
    public:
     AlphaDropout(const double ratio = 0.5,
                  const double alphaDash = -alpha * lambda);
   
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
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     double Ratio() const { return ratio; }
   
     double A() const { return a; }
   
     double B() const { return b; }
   
     double AlphaDash() const {return alphaDash; }
   
     OutputDataType const& Mask() const {return mask;}
   
     void Ratio(const double r)
     {
       ratio = r;
       a = pow((1 - ratio) * (1 + ratio * pow(alphaDash, 2)), -0.5);
       b = -a * alphaDash * ratio;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     OutputDataType mask;
   
     double ratio;
   
     double alphaDash;
   
     bool deterministic;
   
     static constexpr double alpha = 1.6732632423543772848170429916717;
   
     static constexpr double lambda = 1.0507009873554804934193349852946;
   
     double a;
   
     double b;
   }; // class AlphaDropout
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "alpha_dropout_impl.hpp"
   
   #endif
