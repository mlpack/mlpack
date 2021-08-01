
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression_impl.hpp:

Program Listing for File softmax_regression_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_softmax_regression_softmax_regression_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/softmax_regression/softmax_regression_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP
   #define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "softmax_regression.hpp"
   
   namespace mlpack {
   namespace regression {
   
   template<typename OptimizerType>
   SoftmaxRegression::SoftmaxRegression(
       const arma::mat& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses,
       const double lambda,
       const bool fitIntercept,
       OptimizerType optimizer) :
       numClasses(numClasses),
       lambda(lambda),
       fitIntercept(fitIntercept)
   {
     Train(data, labels, numClasses, optimizer);
   }
   
   template<typename OptimizerType, typename... CallbackTypes>
   SoftmaxRegression::SoftmaxRegression(
       const arma::mat& data,
       const arma::Row<size_t>& labels,
       const size_t numClasses,
       const double lambda,
       const bool fitIntercept,
       OptimizerType optimizer,
       CallbackTypes&&... callbacks) :
       numClasses(numClasses),
       lambda(lambda),
       fitIntercept(fitIntercept)
   {
     Train(data, labels, numClasses, optimizer, callbacks...);
   }
   
   template<typename VecType>
   size_t SoftmaxRegression::Classify(const VecType& point) const
   {
     arma::Row<size_t> label(1);
     Classify(point, label);
     return size_t(label(0));
   }
   
   template<typename OptimizerType>
   double SoftmaxRegression::Train(const arma::mat& data,
                                   const arma::Row<size_t>& labels,
                                   const size_t numClasses,
                                   OptimizerType optimizer)
   {
     SoftmaxRegressionFunction regressor(data, labels, numClasses, lambda,
                                         fitIntercept);
     if (parameters.n_elem != regressor.GetInitialPoint().n_elem)
       parameters = regressor.GetInitialPoint();
   
     // Train the model.
     Timer::Start("softmax_regression_optimization");
     const double out = optimizer.Optimize(regressor, parameters);
     Timer::Stop("softmax_regression_optimization");
   
     Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
               << "trained model is " << out << "." << std::endl;
   
     return out;
   }
   
   template<typename OptimizerType, typename... CallbackTypes>
   double SoftmaxRegression::Train(const arma::mat& data,
                                   const arma::Row<size_t>& labels,
                                   const size_t numClasses,
                                   OptimizerType optimizer,
                                   CallbackTypes&&... callbacks)
   {
     SoftmaxRegressionFunction regressor(data, labels, numClasses, lambda,
                                         fitIntercept);
     if (parameters.n_elem != regressor.GetInitialPoint().n_elem)
       parameters = regressor.GetInitialPoint();
   
     // Train the model.
     Timer::Start("softmax_regression_optimization");
     const double out = optimizer.Optimize(regressor, parameters, callbacks...);
     Timer::Stop("softmax_regression_optimization");
   
     Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
               << "trained model is " << out << "." << std::endl;
   
     return out;
   }
   
   } // namespace regression
   } // namespace mlpack
   
   #endif
