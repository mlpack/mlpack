
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder.hpp:

Program Listing for File sparse_autoencoder.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_autoencoder/sparse_autoencoder.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP
   #define MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <ensmallen.hpp>
   
   #include "sparse_autoencoder_function.hpp"
   
   namespace mlpack {
   namespace nn {
   
   class SparseAutoencoder
   {
    public:
     template<typename OptimizerType = ens::L_BFGS>
     SparseAutoencoder(const arma::mat& data,
                       const size_t visibleSize,
                       const size_t hiddenSize,
                       const double lambda = 0.0001,
                       const double beta = 3,
                       const double rho = 0.01,
                       OptimizerType optimizer = OptimizerType());
   
     template<typename OptimizerType, typename... CallbackTypes>
     SparseAutoencoder(const arma::mat& data,
                       const size_t visibleSize,
                       const size_t hiddenSize,
                       const double lambda,
                       const double beta,
                       const double rho ,
                       OptimizerType optimizer,
                       CallbackTypes&&... callbacks);
   
     void GetNewFeatures(arma::mat& data, arma::mat& features);
   
     void Sigmoid(const arma::mat& x, arma::mat& output) const
     {
       output = (1.0 / (1 + arma::exp(-x)));
     }
   
     void VisibleSize(const size_t visible)
     {
       this->visibleSize = visible;
     }
   
     size_t VisibleSize() const
     {
       return visibleSize;
     }
   
     void HiddenSize(const size_t hidden)
     {
       this->hiddenSize = hidden;
     }
   
     size_t HiddenSize() const
     {
       return hiddenSize;
     }
   
     void Lambda(const double l)
     {
       this->lambda = l;
     }
   
     double Lambda() const
     {
       return lambda;
     }
   
     void Beta(const double b)
     {
       this->beta = b;
     }
   
     double Beta() const
     {
       return beta;
     }
   
     void Rho(const double r)
     {
       this->rho = r;
     }
   
     double Rho() const
     {
       return rho;
     }
   
    private:
     arma::mat parameters;
     size_t visibleSize;
     size_t hiddenSize;
     double lambda;
     double beta;
     double rho;
   };
   
   } // namespace nn
   } // namespace mlpack
   
   // Include implementation.
   #include "sparse_autoencoder_impl.hpp"
   
   #endif
