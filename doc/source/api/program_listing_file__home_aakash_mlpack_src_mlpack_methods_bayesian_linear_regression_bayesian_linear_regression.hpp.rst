
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_bayesian_linear_regression_bayesian_linear_regression.hpp:

Program Listing for File bayesian_linear_regression.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_bayesian_linear_regression_bayesian_linear_regression.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/bayesian_linear_regression/bayesian_linear_regression.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_HPP
   #define MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace regression {
   
   class BayesianLinearRegression
   {
    public:
     BayesianLinearRegression(const bool centerData = true,
                              const bool scaleData = false,
                              const size_t maxIterations = 50,
                              const double tolerance = 1e-4);
   
     double Train(const arma::mat& data,
                  const arma::rowvec& responses);
   
     void Predict(const arma::mat& points,
                  arma::rowvec& predictions) const;
   
     void Predict(const arma::mat& points,
                  arma::rowvec& predictions,
                  arma::rowvec& std) const;
   
     double RMSE(const arma::mat& data,
                 const arma::rowvec& responses) const;
   
     const arma::colvec& Omega() const { return omega; }
   
     double Alpha() const { return alpha; }
   
     double Beta() const { return beta; }
   
     double Variance() const { return 1.0 / Beta(); }
   
     const arma::colvec& DataOffset() const { return dataOffset; }
   
     const arma::colvec& DataScale() const { return dataScale; }
   
     double ResponsesOffset() const { return responsesOffset; }
   
     bool CenterData() const { return centerData; }
     bool& CenterData() { return centerData; }
   
     bool ScaleData() const { return scaleData; }
     bool& ScaleData() { return scaleData; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     double Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     bool centerData;
   
     bool scaleData;
   
     size_t maxIterations;
   
     double tolerance;
   
     arma::colvec dataOffset;
   
     arma::colvec dataScale;
   
     double responsesOffset;
   
     double alpha;
   
     double beta;
   
     double gamma;
   
     arma::colvec omega;
   
     arma::mat matCovariance;
   
     double CenterScaleData(const arma::mat& data,
                            const arma::rowvec& responses,
                            arma::mat& dataProc,
                            arma::rowvec& responsesProc);
   
     void CenterScaleDataPred(const arma::mat& data,
                              arma::mat& dataProc) const;
   };
   } // namespace regression
   } // namespace mlpack
   
   // Include implementation of serialize.
   #include "bayesian_linear_regression_impl.hpp"
   
   #endif
