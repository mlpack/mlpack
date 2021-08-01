
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm.hpp:

Program Listing for File hmm.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hmm/hmm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HMM_HMM_HPP
   #define MLPACK_METHODS_HMM_HMM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/discrete_distribution.hpp>
   
   namespace mlpack {
   namespace hmm  {
   
   template<typename Distribution = distribution::DiscreteDistribution>
   class HMM
   {
    public:
     HMM(const size_t states = 0,
         const Distribution emissions = Distribution(),
         const double tolerance = 1e-5);
   
     HMM(const arma::vec& initial,
         const arma::mat& transition,
         const std::vector<Distribution>& emission,
         const double tolerance = 1e-5);
   
     double Train(const std::vector<arma::mat>& dataSeq);
   
     void Train(const std::vector<arma::mat>& dataSeq,
                const std::vector<arma::Row<size_t> >& stateSeq);
   
     double LogEstimate(const arma::mat& dataSeq,
                        arma::mat& stateLogProb,
                        arma::mat& forwardLogProb,
                        arma::mat& backwardLogProb,
                        arma::vec& logScales) const;
   
     double Estimate(const arma::mat& dataSeq,
                     arma::mat& stateProb,
                     arma::mat& forwardProb,
                     arma::mat& backwardProb,
                     arma::vec& scales) const;
   
     double Estimate(const arma::mat& dataSeq,
                     arma::mat& stateProb) const;
   
     void Generate(const size_t length,
                   arma::mat& dataSequence,
                   arma::Row<size_t>& stateSequence,
                   const size_t startState = 0) const;
   
     double Predict(const arma::mat& dataSeq,
                    arma::Row<size_t>& stateSeq) const;
   
     double LogLikelihood(const arma::mat& dataSeq) const;
   
     double EmissionLogScaleFactor(const arma::vec& emissionLogProb,
                                   arma::vec& forwardLogProb) const;
   
     double EmissionLogLikelihood(const arma::vec& emissionLogProb,
                                  double &logLikelihood,
                                  arma::vec& forwardLogProb) const;
   
     double LogScaleFactor(const arma::vec &data,
                           arma::vec& forwardLogProb) const;
   
     double LogLikelihood(const arma::vec &data,
                          double &logLikelihood,
                          arma::vec& forwardLogProb) const;
     void Filter(const arma::mat& dataSeq,
                 arma::mat& filterSeq,
                 size_t ahead = 0) const;
   
     void Smooth(const arma::mat& dataSeq,
                 arma::mat& smoothSeq) const;
   
     const arma::vec& Initial() const { return initialProxy; }
     arma::vec& Initial()
     {
       recalculateInitial = true;
       return initialProxy;
     }
   
     const arma::mat& Transition() const { return transitionProxy; }
     arma::mat& Transition()
     {
       recalculateTransition = true;
       return transitionProxy;
     }
   
     const std::vector<Distribution>& Emission() const { return emission; }
     std::vector<Distribution>& Emission() { return emission; }
   
     size_t Dimensionality() const { return dimensionality; }
     size_t& Dimensionality() { return dimensionality; }
   
     double Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
     template<typename Archive>
     void load(Archive& ar, const uint32_t version);
   
     template<typename Archive>
     void save(Archive& ar, const uint32_t version) const;
   
    protected:
     arma::vec ForwardAtT0(const arma::vec& emissionLogProb,
                           double& logScales) const;
   
     arma::vec ForwardAtTn(const arma::vec& emissionLogProb,
                           double& logScales,
                           const arma::vec& prevForwardLogProb) const;
   
     // Helper functions.
     void Forward(const arma::mat& dataSeq,
                  arma::vec& logScales,
                  arma::mat& forwardLogProb,
                  arma::mat& logProbs) const;
   
     void Backward(const arma::mat& dataSeq,
                   const arma::vec& logScales,
                   arma::mat& backwardLogProb,
                   arma::mat& logProbs) const;
   
     std::vector<Distribution> emission;
   
     arma::mat transitionProxy;
   
     mutable arma::mat logTransition;
   
    private:
     void ConvertToLogSpace() const;
   
     arma::vec initialProxy;
   
     mutable arma::vec logInitial;
   
     size_t dimensionality;
   
     double tolerance;
   
     mutable bool recalculateInitial;
   
     mutable bool recalculateTransition;
   };
   
   } // namespace hmm
   } // namespace mlpack
   
   // Include implementation.
   #include "hmm_impl.hpp"
   
   #endif
