
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_regression.hpp:

Program Listing for File hmm_regression.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_regression.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hmm/hmm_regression.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HMM_HMM_REGRESSION_HPP
   #define MLPACK_METHODS_HMM_HMM_REGRESSION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/regression_distribution.hpp>
   #include "hmm.hpp"
   
   namespace mlpack {
   namespace hmm  {
   
   class HMMRegression : public HMM<distribution::RegressionDistribution>
   {
    public:
     HMMRegression(const size_t states,
         const distribution::RegressionDistribution emissions,
         const double tolerance = 1e-5) :
         HMM<distribution::RegressionDistribution>(states, emissions, tolerance)
     { /* nothing to do */ }
   
     HMMRegression(const arma::vec& initial,
         const arma::mat& transition,
         const std::vector<distribution::RegressionDistribution>& emission,
         const double tolerance = 1e-5) :
         HMM<distribution::RegressionDistribution>(initial, transition, emission,
          tolerance)
     { /* nothing to do */ }
   
   
   
     void Train(const std::vector<arma::mat>& predictors,
                const std::vector<arma::vec>& responses);
   
     void Train(const std::vector<arma::mat>& predictors,
                const std::vector<arma::vec>& responses,
                const std::vector<arma::Row<size_t> >& stateSeq);
   
     double Estimate(const arma::mat& predictors,
                     const arma::vec& responses,
                     arma::mat& stateProb,
                     arma::mat& forwardProb,
                     arma::mat& backwardProb,
                     arma::vec& scales) const;
   
     double Estimate(const arma::mat& predictors,
                     const arma::vec& responses,
                     arma::mat& stateProb) const;
   
     double Predict(const arma::mat& predictors,
                    const arma::vec& responses,
                    arma::Row<size_t>& stateSeq) const;
   
     double LogLikelihood(const arma::mat& predictors,
                          const arma::vec& responses) const;
     void Filter(const arma::mat& predictors,
                 const arma::vec& responses,
                 arma::vec& filterSeq,
                 size_t ahead = 0) const;
   
     void Smooth(const arma::mat& predictors,
                 const arma::vec& responses,
                 arma::vec& smoothSeq) const;
   
    private:
     void StackData(const std::vector<arma::mat>& predictors,
                    const std::vector<arma::vec>& responses,
                    std::vector<arma::mat>& dataSeq) const;
   
     void StackData(const arma::mat& predictors,
                    const arma::vec& responses,
                    arma::mat& dataSeq) const;
   
     void Forward(const arma::mat& predictors,
                  const arma::vec& responses,
                  arma::vec& scales,
                  arma::mat& forwardProb) const;
   
     void Backward(const arma::mat& predictors,
                   const arma::vec& responses,
                   const arma::vec& scales,
                   arma::mat& backwardProb) const;
   };
   
   } // namespace hmm
   } // namespace mlpack
   
   // Include implementation.
   #include "hmm_regression_impl.hpp"
   
   #endif
