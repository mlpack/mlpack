
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder_function.hpp:

Program Listing for File sparse_autoencoder_function.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder_function.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_autoencoder/sparse_autoencoder_function.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_FUNCTION_HPP
   #define MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_FUNCTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace nn {
   
   class SparseAutoencoderFunction
   {
    public:
     SparseAutoencoderFunction(const arma::mat& data,
                               const size_t visibleSize,
                               const size_t hiddenSize,
                               const double lambda = 0.0001,
                               const double beta = 3,
                               const double rho = 0.01);
   
     const arma::mat InitializeWeights();
   
     double Evaluate(const arma::mat& parameters) const;
   
     void Gradient(const arma::mat& parameters, arma::mat& gradient) const;
   
     void Sigmoid(const arma::mat& x, arma::mat& output) const
     {
       output = (1.0 / (1 + arma::exp(-x)));
     }
   
     const arma::mat& GetInitialPoint() const { return initialPoint; }
   
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
     const arma::mat& data;
     arma::mat initialPoint;
     size_t visibleSize;
     size_t hiddenSize;
     double lambda;
     double beta;
     double rho;
   };
   
   } // namespace nn
   } // namespace mlpack
   
   #endif
