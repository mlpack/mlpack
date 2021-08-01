
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder_impl.hpp:

Program Listing for File sparse_autoencoder_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_autoencoder/sparse_autoencoder_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_IMPL_HPP
   #define MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "sparse_autoencoder.hpp"
   
   namespace mlpack {
   namespace nn {
   
   template<typename OptimizerType>
   SparseAutoencoder::SparseAutoencoder(const arma::mat& data,
                                        const size_t visibleSize,
                                        const size_t hiddenSize,
                                        double lambda,
                                        double beta,
                                        double rho,
                                        OptimizerType optimizer) :
       visibleSize(visibleSize),
       hiddenSize(hiddenSize),
       lambda(lambda),
       beta(beta),
       rho(rho)
   {
     SparseAutoencoderFunction encoderFunction(data, visibleSize, hiddenSize,
                                               lambda, beta, rho);
   
     parameters = encoderFunction.GetInitialPoint();
   
     // Train the model.
     Timer::Start("sparse_autoencoder_optimization");
     const double out = optimizer.Optimize(encoderFunction, parameters);
     Timer::Stop("sparse_autoencoder_optimization");
   
     Log::Info << "SparseAutoencoder::SparseAutoencoder(): final objective of "
         << "trained model is " << out << "." << std::endl;
   }
   
   template<typename OptimizerType, typename... CallbackTypes>
   SparseAutoencoder::SparseAutoencoder(const arma::mat& data,
                                        const size_t visibleSize,
                                        const size_t hiddenSize,
                                        double lambda,
                                        double beta,
                                        double rho,
                                        OptimizerType optimizer,
                                        CallbackTypes&&... callbacks) :
       visibleSize(visibleSize),
       hiddenSize(hiddenSize),
       lambda(lambda),
       beta(beta),
       rho(rho)
   {
     SparseAutoencoderFunction encoderFunction(data, visibleSize, hiddenSize,
                                               lambda, beta, rho);
   
     parameters = encoderFunction.GetInitialPoint();
   
     // Train the model.
     Timer::Start("sparse_autoencoder_optimization");
     const double out = optimizer.Optimize(encoderFunction, parameters,
         callbacks...);
     Timer::Stop("sparse_autoencoder_optimization");
   
     Log::Info << "SparseAutoencoder::SparseAutoencoder(): final objective of "
         << "trained model is " << out << "." << std::endl;
   }
   
   } // namespace nn
   } // namespace mlpack
   
   #endif
