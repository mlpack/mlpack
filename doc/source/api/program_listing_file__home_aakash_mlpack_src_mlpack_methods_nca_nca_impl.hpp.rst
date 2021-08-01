
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_nca_nca_impl.hpp:

Program Listing for File nca_impl.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_nca_nca_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/nca/nca_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NCA_NCA_IMPL_HPP
   #define MLPACK_METHODS_NCA_NCA_IMPL_HPP
   
   // In case it was not already included.
   #include "nca.hpp"
   
   namespace mlpack {
   namespace nca {
   
   // Just set the internal matrix reference.
   template<typename MetricType, typename OptimizerType>
   NCA<MetricType, OptimizerType>::NCA(const arma::mat& dataset,
                                       const arma::Row<size_t>& labels,
                                       MetricType metric) :
       dataset(dataset),
       labels(labels),
       metric(metric),
       errorFunction(dataset, labels, metric)
   { /* Nothing to do. */ }
   
   template<typename MetricType, typename OptimizerType>
   template<typename... CallbackTypes>
   void NCA<MetricType, OptimizerType>::LearnDistance(arma::mat& outputMatrix,
       CallbackTypes&&... callbacks)
   {
     // See if we were passed an initialized matrix.
     if ((outputMatrix.n_rows != dataset.n_rows) ||
         (outputMatrix.n_cols != dataset.n_rows))
       outputMatrix.eye(dataset.n_rows, dataset.n_rows);
   
     Timer::Start("nca_sgd_optimization");
   
     optimizer.Optimize(errorFunction, outputMatrix, callbacks...);
   
     Timer::Stop("nca_sgd_optimization");
   }
   
   } // namespace nca
   } // namespace mlpack
   
   #endif
