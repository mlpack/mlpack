
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_viterbi_main.cpp:

Program Listing for File hmm_viterbi_main.cpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_viterbi_main.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/hmm/hmm_viterbi_main.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/io.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   
   #include "hmm.hpp"
   #include "hmm_model.hpp"
   
   #include <mlpack/methods/gmm/gmm.hpp>
   #include <mlpack/methods/gmm/diagonal_gmm.hpp>
   
   using namespace mlpack;
   using namespace mlpack::hmm;
   using namespace mlpack::distribution;
   using namespace mlpack::util;
   using namespace mlpack::gmm;
   using namespace arma;
   using namespace std;
   
   // Program Name.
   BINDING_NAME("Hidden Markov Model (HMM) Viterbi State Prediction");
   
   // Short description.
   BINDING_SHORT_DESC(
       "A utility for computing the most probable hidden state sequence for Hidden"
       " Markov Models (HMMs).  Given a pre-trained HMM and an observed sequence, "
       "this uses the Viterbi algorithm to compute and return the most probable "
       "hidden state sequence.");
   
   // Long description.
   BINDING_LONG_DESC(
       "This utility takes an already-trained HMM, specified as " +
       PRINT_PARAM_STRING("input_model") + ", and evaluates the most probable "
       "hidden state sequence of a given sequence of observations (specified as "
       "'" + PRINT_PARAM_STRING("input") + ", using the Viterbi algorithm.  The "
       "computed state sequence may be saved using the " +
       PRINT_PARAM_STRING("output") + " output parameter.");
   
   // Example.
   BINDING_EXAMPLE(
       "For example, to predict the state sequence of the observations " +
       PRINT_DATASET("obs") + " using the HMM " + PRINT_MODEL("hmm") + ", "
       "storing the predicted state sequence to " + PRINT_DATASET("states") +
       ", the following command could be used:"
       "\n\n" +
       PRINT_CALL("hmm_viterbi", "input", "obs", "input_model", "hmm", "output",
           "states"));
   
   // See also...
   BINDING_SEE_ALSO("@hmm_train", "#hmm_train");
   BINDING_SEE_ALSO("@hmm_generate", "#hmm_generate");
   BINDING_SEE_ALSO("@hmm_loglik", "#hmm_loglik");
   BINDING_SEE_ALSO("Hidden Mixture Models on Wikipedia",
           "https://en.wikipedia.org/wiki/Hidden_Markov_model");
   BINDING_SEE_ALSO("mlpack::hmm::HMM class documentation",
           "@doxygen/classmlpack_1_1hmm_1_1HMM.html");
   
   PARAM_MATRIX_IN_REQ("input", "Matrix containing observations,", "i");
   PARAM_MODEL_IN_REQ(HMMModel, "input_model", "Trained HMM to use.", "m");
   PARAM_UMATRIX_OUT("output", "File to save predicted state sequence to.", "o");
   
   // Because we don't know what the type of our HMM is, we need to write a
   // function that can take arbitrary HMM types.
   struct Viterbi
   {
     template<typename HMMType>
     static void Apply(HMMType& hmm, void* /* extraInfo */)
     {
       // Load observations.
       mat dataSeq = std::move(IO::GetParam<arma::mat>("input"));
   
       // See if transposing the data could make it the right dimensionality.
       if ((dataSeq.n_cols == 1) && (hmm.Emission()[0].Dimensionality() == 1))
       {
         Log::Info << "Data sequence appears to be transposed; correcting."
             << endl;
         dataSeq = dataSeq.t();
       }
   
       // Verify correct dimensionality.
       if (dataSeq.n_rows != hmm.Emission()[0].Dimensionality())
       {
         Log::Fatal << "Observation dimensionality (" << dataSeq.n_rows << ") "
             << "does not match HMM Gaussian dimensionality ("
             << hmm.Emission()[0].Dimensionality() << ")!" << endl;
       }
   
       arma::Row<size_t> sequence;
       hmm.Predict(dataSeq, sequence);
   
       // Save output.
       IO::GetParam<arma::Mat<size_t>>("output") = std::move(sequence);
     }
   };
   
   static void mlpackMain()
   {
     RequireAtLeastOnePassed({ "output" }, false, "no results will be saved");
   
     IO::GetParam<HMMModel*>("input_model")->PerformAction<Viterbi>((void*) NULL);
   }
