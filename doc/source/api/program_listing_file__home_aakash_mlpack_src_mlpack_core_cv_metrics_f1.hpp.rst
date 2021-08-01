
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_metrics_f1.hpp:

Program Listing for File f1.hpp
===============================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_metrics_f1.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/metrics/f1.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_METRICS_F1_HPP
   #define MLPACK_CORE_CV_METRICS_F1_HPP
   
   #include <type_traits>
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/cv/metrics/average_strategy.hpp>
   
   namespace mlpack {
   namespace cv {
   
   template<AverageStrategy AS, size_t PositiveClass = 1>
   class F1
   {
    public:
     template<typename MLAlgorithm, typename DataType>
     static double Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const arma::Row<size_t>& labels);
   
     static const bool NeedsMinimization = false;
   
    private:
     template<AverageStrategy _AS,
              typename MLAlgorithm,
              typename DataType,
              typename = std::enable_if_t<_AS == Binary>>
     static double Evaluate(MLAlgorithm& model,
                           const DataType& data,
                           const arma::Row<size_t>& labels);
   
     template<AverageStrategy _AS,
              typename MLAlgorithm,
              typename DataType,
              typename = std::enable_if_t<_AS == Micro>,
              typename = void>
     static double Evaluate(MLAlgorithm& model,
                           const DataType& data,
                           const arma::Row<size_t>& labels);
   
     template<AverageStrategy _AS,
              typename MLAlgorithm,
              typename DataType,
              typename = std::enable_if_t<_AS == Macro>,
              typename = void,
              typename = void>
     static double Evaluate(MLAlgorithm& model,
                           const DataType& data,
                           const arma::Row<size_t>& labels);
   };
   
   } // namespace cv
   } // namespace mlpack
   
   // Include implementation.
   #include "f1_impl.hpp"
   
   #endif
