
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_regression_impl.hpp:

Program Listing for File hmm_regression_impl.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_regression_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hmm/hmm_regression_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HMM_HMM_REGRESSION_IMPL_HPP
   #define MLPACK_METHODS_HMM_HMM_REGRESSION_IMPL_HPP
   
   // Just in case...
   #include "hmm_regression.hpp"
   
   namespace mlpack {
   namespace hmm {
   
   void HMMRegression::Train(const std::vector<arma::mat>& predictors,
                             const std::vector<arma::vec>& responses)
   {
     std::vector<arma::mat> dataSeq;
     StackData(predictors, responses, dataSeq);
     this->HMM::Train(dataSeq);
   }
   
   void HMMRegression::Train(const std::vector<arma::mat>& predictors,
                             const std::vector<arma::vec>& responses,
                             const std::vector<arma::Row<size_t> >& stateSeq)
   {
     std::vector<arma::mat> dataSeq;
     StackData(predictors, responses, dataSeq);
     this->HMM::Train(dataSeq, stateSeq);
   }
   
   double HMMRegression::Estimate(const arma::mat& predictors,
                                  const arma::vec& responses,
                                  arma::mat& stateProb,
                                  arma::mat& forwardProb,
                                  arma::mat& backwardProb,
                                  arma::vec& scales) const
   {
     arma::mat dataSeq;
     StackData(predictors, responses, dataSeq);
     return this->HMM::Estimate(dataSeq, stateProb, forwardProb,
         backwardProb, scales);
   }
   
   double HMMRegression::Estimate(const arma::mat& predictors,
                                  const arma::vec& responses,
                                  arma::mat& stateProb) const
   {
     arma::mat dataSeq;
     StackData(predictors, responses, dataSeq);
     return this->HMM::Estimate(dataSeq, stateProb);
   }
   
   double HMMRegression::Predict(const arma::mat& predictors,
                                 const arma::vec& responses,
                                 arma::Row<size_t>& stateSeq) const
   {
     arma::mat dataSeq;
     StackData(predictors, responses, dataSeq);
     return this->HMM::Predict(dataSeq, stateSeq);
   }
   
   double HMMRegression::LogLikelihood(const arma::mat& predictors,
                                       const arma::vec& responses) const
   {
     arma::mat dataSeq;
     StackData(predictors, responses, dataSeq);
     return this->HMM::LogLikelihood(dataSeq);
   }
   
   void HMMRegression::Filter(const arma::mat& predictors,
                              const arma::vec& responses,
                              arma::vec& filterSeq,
                              size_t ahead) const
   {
     // First run the forward algorithm
     arma::mat forwardProb;
     arma::vec scales;
     Forward(predictors, responses, scales, forwardProb);
   
     // Propagate state, predictors ahead
     if (ahead != 0)
     {
       forwardProb = pow(transition, ahead)*forwardProb;
       forwardProb = forwardProb.cols(0, forwardProb.n_cols-ahead-1);
     }
   
     // Compute expected emissions.
     filterSeq.resize(responses.n_elem - ahead);
     filterSeq.zeros();
     arma::vec nextSeq;
     for (size_t i = 0; i < emission.size(); ++i)
     {
       emission[i].Predict(predictors.cols(ahead, predictors.n_cols-1), nextSeq);
       filterSeq = filterSeq + nextSeq%(forwardProb.row(i).t());
     }
   }
   
   void HMMRegression::Smooth(const arma::mat& predictors,
                              const arma::vec& responses,
                              arma::vec& smoothSeq) const
   {
     // First run the forward algorithm
     arma::mat stateProb;
     Estimate(predictors, responses, stateProb);
   
     // Compute expected emissions.
     smoothSeq.resize(responses.n_elem);
     smoothSeq.zeros();
     arma::vec nextSeq;
     for (size_t i = 0; i < emission.size(); ++i)
     {
       emission[i].Predict(predictors, nextSeq);
       smoothSeq = smoothSeq + nextSeq%(stateProb.row(i).t());
     }
   }
   
   void HMMRegression::Forward(const arma::mat& predictors,
                               const arma::vec& responses,
                               arma::vec& scales,
                               arma::mat& forwardProb) const
   {
     arma::mat dataSeq;
     StackData(predictors, responses, dataSeq);
     this->HMM::Forward(dataSeq, scales, forwardProb);
   }
   
   
   void HMMRegression::Backward(const arma::mat& predictors,
                                const arma::vec& responses,
                                const arma::vec& scales,
                                arma::mat& backwardProb) const
   {
     arma::mat dataSeq;
     StackData(predictors, responses, dataSeq);
     this->HMM::Backward(dataSeq, scales, backwardProb);
   }
   
   void HMMRegression::StackData(const std::vector<arma::mat>& predictors,
                                 const std::vector<arma::vec>& responses,
                                 std::vector<arma::mat>& dataSeq) const
   {
     arma::mat nextSeq;
     for (size_t i = 0; i < predictors.size(); ++i)
     {
       nextSeq = predictors[i];
       nextSeq.insert_rows(0, responses[i].t());
       dataSeq.push_back(nextSeq);
     }
   }
   
   void HMMRegression::StackData(const arma::mat& predictors,
                                 const arma::vec& responses,
                                 arma::mat& dataSeq) const
   {
     dataSeq = predictors;
     dataSeq.insert_rows(0, responses.t());
   }
   
   } // namespace hmm
   } // namespace mlpack
   
   #endif
