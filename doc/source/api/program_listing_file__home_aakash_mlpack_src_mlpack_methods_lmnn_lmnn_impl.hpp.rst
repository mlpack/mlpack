
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_lmnn_lmnn_impl.hpp:

Program Listing for File lmnn_impl.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_lmnn_lmnn_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/lmnn/lmnn_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LMNN_LMNN_IMPL_HPP
   #define MLPACK_METHODS_LMNN_LMNN_IMPL_HPP
   
   // In case it was not already included.
   #include "lmnn.hpp"
   
   namespace mlpack {
   namespace lmnn {
   
   template<typename MetricType, typename OptimizerType>
   LMNN<MetricType, OptimizerType>::LMNN(const arma::mat& dataset,
                          const arma::Row<size_t>& labels,
                          const size_t k,
                          const MetricType metric) :
       dataset(dataset),
       labels(labels),
       k(k),
       regularization(0.5),
       range(1),
       metric(metric)
   { /* nothing to do */ }
   
   template<typename MetricType, typename OptimizerType>
   template<typename... CallbackTypes>
   void LMNN<MetricType, OptimizerType>::LearnDistance(arma::mat& outputMatrix,
       CallbackTypes&&... callbacks)
   {
     // LMNN objective function.
     LMNNFunction<MetricType> objFunction(dataset, labels, k,
         regularization, range);
   
     // See if we were passed an initialized matrix. outputMatrix (L) must be
     // having r x d dimensionality.
     if ((outputMatrix.n_cols != dataset.n_rows) ||
         (outputMatrix.n_rows > dataset.n_rows) ||
         !(arma::is_finite(outputMatrix)))
     {
       Log::Info << "Initial learning point have invalid dimensionality.  "
           "Identity matrix will be used as initial learning point for "
            "optimization." << std::endl;
       outputMatrix.eye(dataset.n_rows, dataset.n_rows);
     }
   
     Timer::Start("lmnn_optimization");
   
     optimizer.Optimize(objFunction, outputMatrix, callbacks...);
   
     Timer::Stop("lmnn_optimization");
   }
   
   
   } // namespace lmnn
   } // namespace mlpack
   
   #endif
