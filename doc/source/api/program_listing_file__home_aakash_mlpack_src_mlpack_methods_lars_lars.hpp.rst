
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_lars_lars.hpp:

Program Listing for File lars.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_lars_lars.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/lars/lars.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LARS_LARS_HPP
   #define MLPACK_METHODS_LARS_LARS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace regression {
   
   // beta is the estimator
   // yHat is the prediction from the current estimator
   
   class LARS
   {
    public:
     LARS(const bool useCholesky = false,
          const double lambda1 = 0.0,
          const double lambda2 = 0.0,
          const double tolerance = 1e-16);
   
     LARS(const bool useCholesky,
          const arma::mat& gramMatrix,
          const double lambda1 = 0.0,
          const double lambda2 = 0.0,
          const double tolerance = 1e-16);
   
     LARS(const arma::mat& data,
          const arma::rowvec& responses,
          const bool transposeData = true,
          const bool useCholesky = false,
          const double lambda1 = 0.0,
          const double lambda2 = 0.0,
          const double tolerance = 1e-16);
   
     LARS(const arma::mat& data,
          const arma::rowvec& responses,
          const bool transposeData,
          const bool useCholesky,
          const arma::mat& gramMatrix,
          const double lambda1 = 0.0,
          const double lambda2 = 0.0,
          const double tolerance = 1e-16);
   
     LARS(const LARS& other);
   
     LARS(LARS&& other);
   
     LARS& operator=(const LARS& other);
   
     LARS& operator=(LARS&& other);
   
     double Train(const arma::mat& data,
                  const arma::rowvec& responses,
                  arma::vec& beta,
                  const bool transposeData = true);
   
     double Train(const arma::mat& data,
                  const arma::rowvec& responses,
                  const bool transposeData = true);
   
     void Predict(const arma::mat& points,
                  arma::rowvec& predictions,
                  const bool rowMajor = false) const;
   
     double Lambda1() const { return lambda1; }
     double& Lambda1() { return lambda1; }
   
     double Lambda2() const { return lambda2; }
     double& Lambda2() { return lambda2; }
   
     bool UseCholesky() const { return useCholesky; }
     bool& UseCholesky() { return useCholesky; }
   
     double Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
     const std::vector<size_t>& ActiveSet() const { return activeSet; }
   
     const std::vector<arma::vec>& BetaPath() const { return betaPath; }
   
     const arma::vec& Beta() const { return betaPath.back(); }
   
     const std::vector<double>& LambdaPath() const { return lambdaPath; }
   
     const arma::mat& MatUtriCholFactor() const { return matUtriCholFactor; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     double ComputeError(const arma::mat& matX,
                         const arma::rowvec& y,
                         const bool rowMajor = false);
   
    private:
     arma::mat matGramInternal;
   
     const arma::mat* matGram;
   
     arma::mat matUtriCholFactor;
   
     bool useCholesky;
   
     bool lasso;
     double lambda1;
   
     bool elasticNet;
     double lambda2;
   
     double tolerance;
   
     std::vector<arma::vec> betaPath;
   
     std::vector<double> lambdaPath;
   
     std::vector<size_t> activeSet;
   
     std::vector<bool> isActive;
   
     // Set of variables that are ignored (if any).
   
     std::vector<size_t> ignoreSet;
   
     std::vector<bool> isIgnored;
   
     void Deactivate(const size_t activeVarInd);
   
     void Activate(const size_t varInd);
   
     void Ignore(const size_t varInd);
   
     // compute "equiangular" direction in output space
     void ComputeYHatDirection(const arma::mat& matX,
                               const arma::vec& betaDirection,
                               arma::vec& yHatDirection);
   
     // interpolate to compute last solution vector
     void InterpolateBeta();
   
     void CholeskyInsert(const arma::vec& newX, const arma::mat& X);
   
     void CholeskyInsert(double sqNormNewX, const arma::vec& newGramCol);
   
     void GivensRotate(const arma::vec::fixed<2>& x,
                       arma::vec::fixed<2>& rotatedX,
                       arma::mat& G);
   
     void CholeskyDelete(const size_t colToKill);
   };
   
   } // namespace regression
   } // namespace mlpack
   
   // Include implementation of serialize().
   #include "lars_impl.hpp"
   
   #endif
