
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_dists_regression_distribution.hpp:

Program Listing for File regression_distribution.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_dists_regression_distribution.hpp>` (``/home/aakash/mlpack/src/mlpack/core/dists/regression_distribution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_HPP
   #define MLPACK_CORE_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/gaussian_distribution.hpp>
   #include <mlpack/methods/linear_regression/linear_regression.hpp>
   
   namespace mlpack {
   namespace distribution {
   
   class RegressionDistribution
   {
    private:
     regression::LinearRegression rf;
     GaussianDistribution err;
   
    public:
     RegressionDistribution() { /* nothing to do */ }
   
     mlpack_deprecated RegressionDistribution(const arma::mat& predictors,
                                              const arma::vec& responses) :
       RegressionDistribution(predictors, arma::rowvec(responses.t()))
     {}
   
     RegressionDistribution(const arma::mat& predictors,
                            const arma::rowvec& responses)
     {
       rf.Train(predictors, responses);
       err = GaussianDistribution(1);
       arma::mat cov(1, 1);
       cov(0, 0) = rf.ComputeError(predictors, responses);
       err.Covariance(std::move(cov));
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(rf));
       ar(CEREAL_NVP(err));
     }
   
     const regression::LinearRegression& Rf() const { return rf; }
     regression::LinearRegression& Rf() { return rf; }
   
     const GaussianDistribution& Err() const { return err; }
     GaussianDistribution& Err() { return err; }
   
     void Train(const arma::mat& observations);
   
     mlpack_deprecated void Train(const arma::mat& observations,
                                  const arma::vec& weights);
   
     void Train(const arma::mat& observations, const arma::rowvec& weights);
   
     double Probability(const arma::vec& observation) const;
   
     double LogProbability(const arma::vec& observation) const
     {
       return log(Probability(observation));
     }
   
     mlpack_deprecated void Predict(const arma::mat& points,
                                    arma::vec& predictions) const;
   
     void Predict(const arma::mat& points, arma::rowvec& predictions) const;
   
     const arma::vec& Parameters() const { return rf.Parameters(); }
   
     size_t Dimensionality() const { return rf.Parameters().n_elem; }
   };
   
   
   } // namespace distribution
   } // namespace mlpack
   
   #endif
