
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_perceptron_perceptron.hpp:

Program Listing for File perceptron.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_perceptron_perceptron.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/perceptron/perceptron.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_PERCEPTRON_PERCEPTRON_HPP
   #define MLPACK_METHODS_PERCEPTRON_PERCEPTRON_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "initialization_methods/zero_init.hpp"
   #include "initialization_methods/random_init.hpp"
   #include "learning_policies/simple_weight_update.hpp"
   
   namespace mlpack {
   namespace perceptron {
   
   template<typename LearnPolicy = SimpleWeightUpdate,
            typename WeightInitializationPolicy = ZeroInitialization,
            typename MatType = arma::mat>
   class Perceptron
   {
    public:
     Perceptron(const size_t numClasses = 0,
                const size_t dimensionality = 0,
                const size_t maxIterations = 1000);
   
     Perceptron(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t maxIterations = 1000);
   
     Perceptron(const Perceptron& other,
                const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const arma::rowvec& instanceWeights);
   
     void Train(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const arma::rowvec& instanceWeights = arma::rowvec());
   
     void Classify(const MatType& test, arma::Row<size_t>& predictedLabels);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     size_t NumClasses() const { return weights.n_cols; }
   
     const arma::mat& Weights() const { return weights; }
     arma::mat& Weights() { return weights; }
   
     const arma::vec& Biases() const { return biases; }
     arma::vec& Biases() { return biases; }
   
    private:
     size_t maxIterations;
   
     arma::mat weights;
   
     arma::vec biases;
   };
   
   } // namespace perceptron
   } // namespace mlpack
   
   #include "perceptron_impl.hpp"
   
   #endif
