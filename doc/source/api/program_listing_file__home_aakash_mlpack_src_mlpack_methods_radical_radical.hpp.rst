
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_radical_radical.hpp:

Program Listing for File radical.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_radical_radical.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/radical/radical.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RADICAL_RADICAL_HPP
   #define MLPACK_METHODS_RADICAL_RADICAL_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace radical {
   
   class Radical
   {
    public:
     Radical(const double noiseStdDev = 0.175,
             const size_t replicates = 30,
             const size_t angles = 150,
             const size_t sweeps = 0,
             const size_t m = 0);
   
     void DoRadical(const arma::mat& matX, arma::mat& matY, arma::mat& matW);
   
     double Vasicek(arma::vec& x) const;
   
     void CopyAndPerturb(arma::mat& xNew, const arma::mat& x) const;
   
     double DoRadical2D(const arma::mat& matX);
   
     double NoiseStdDev() const { return noiseStdDev; }
     double& NoiseStdDev() { return noiseStdDev; }
   
     size_t Replicates() const { return replicates; }
     size_t& Replicates() { return replicates; }
   
     size_t Angles() const { return angles; }
     size_t& Angles() { return angles; }
   
     size_t Sweeps() const { return sweeps; }
     size_t& Sweeps() { return sweeps; }
   
    private:
     double noiseStdDev;
   
     size_t replicates;
   
     size_t angles;
   
     size_t sweeps;
   
     size_t m;
   
     arma::mat perturbed;
     arma::mat candidate;
   };
   
   void WhitenFeatureMajorMatrix(const arma::mat& matX,
                                 arma::mat& matXWhitened,
                                 arma::mat& matWhitening);
   
   } // namespace radical
   } // namespace mlpack
   
   #endif
