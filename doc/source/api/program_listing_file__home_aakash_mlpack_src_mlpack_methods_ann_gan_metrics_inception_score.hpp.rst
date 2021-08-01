
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_gan_metrics_inception_score.hpp:

Program Listing for File inception_score.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_gan_metrics_inception_score.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/gan/metrics/inception_score.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_METRICS_INCEPTION_SCORE_HPP
   #define MLPACK_METHODS_METRICS_INCEPTION_SCORE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann /* Artificial Neural Network */ {
   
   template<typename ModelType>
   double InceptionScore(ModelType Model,
                         arma::mat images,
                         size_t splits = 1);
   
   
   } // namespace ann
   } // namespace mlpack
   
   #include "inception_score_impl.hpp"
   
   #endif
