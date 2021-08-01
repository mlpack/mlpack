
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reparametrization.hpp:

Program Listing for File reparametrization.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_reparametrization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/reparametrization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_REPARAMETRIZATION_HPP
   #define MLPACK_METHODS_ANN_LAYER_REPARAMETRIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "layer_types.hpp"
   #include "../activation_functions/softplus_function.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Reparametrization
   {
    public:
     Reparametrization();
   
     Reparametrization(const size_t latentSize,
                       const bool stochastic = true,
                       const bool includeKl = true,
                       const double beta = 1);
   
     Reparametrization(const Reparametrization& layer);
   
     Reparametrization(Reparametrization&& layer);
   
     Reparametrization& operator=(const Reparametrization& layer);
   
     Reparametrization& operator=(Reparametrization&& layer);
   
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
   
     size_t const& OutputSize() const { return latentSize; }
     size_t& OutputSize() { return latentSize; }
   
     double Loss()
     {
       if (!includeKl)
         return 0;
   
       return -0.5 * beta * arma::accu(2 * arma::log(stdDev) - arma::pow(stdDev, 2)
           - arma::pow(mean, 2) + 1) / mean.n_cols;
     }
   
     bool Stochastic() const { return stochastic; }
   
     bool IncludeKL() const { return includeKl; }
   
     double Beta() const { return beta; }
   
     size_t InputShape() const
     {
       return 2 * latentSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t latentSize;
   
     bool stochastic;
   
     bool includeKl;
   
     double beta;
   
     OutputDataType delta;
   
     OutputDataType gaussianSample;
   
     OutputDataType mean;
   
     OutputDataType preStdDev;
   
     OutputDataType stdDev;
   
     OutputDataType outputParameter;
   }; // class Reparametrization
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "reparametrization_impl.hpp"
   
   #endif
