
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_local_coordinate_coding_lcc.hpp:

Program Listing for File lcc.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_local_coordinate_coding_lcc.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/local_coordinate_coding/lcc.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_HPP
   #define MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/lars/lars.hpp>
   
   // Include three simple dictionary initializers from sparse coding.
   #include "../sparse_coding/nothing_initializer.hpp"
   #include "../sparse_coding/data_dependent_random_initializer.hpp"
   #include "../sparse_coding/random_initializer.hpp"
   
   namespace mlpack {
   namespace lcc {
   
   class LocalCoordinateCoding
   {
    public:
     template<
         typename DictionaryInitializer =
             sparse_coding::DataDependentRandomInitializer
     >
     LocalCoordinateCoding(const arma::mat& data,
                           const size_t atoms,
                           const double lambda,
                           const size_t maxIterations = 0,
                           const double tolerance = 0.01,
                           const DictionaryInitializer& initializer =
                               DictionaryInitializer());
   
     LocalCoordinateCoding(const size_t atoms = 0,
                           const double lambda = 0.0,
                           const size_t maxIterations = 0,
                           const double tolerance = 0.01);
   
     template<
         typename DictionaryInitializer =
             sparse_coding::DataDependentRandomInitializer
     >
     double Train(const arma::mat& data,
                  const DictionaryInitializer& initializer =
                      DictionaryInitializer());
   
     void Encode(const arma::mat& data, arma::mat& codes);
   
     void OptimizeDictionary(const arma::mat& data,
                             const arma::mat& codes,
                             const arma::uvec& adjacencies);
   
     double Objective(const arma::mat& data,
                      const arma::mat& codes,
                      const arma::uvec& adjacencies) const;
   
     size_t Atoms() const { return atoms; }
     size_t& Atoms() { return atoms; }
   
     const arma::mat& Dictionary() const { return dictionary; }
     arma::mat& Dictionary() { return dictionary; }
   
     double Lambda() const { return lambda; }
     double& Lambda() { return lambda; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     double Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t atoms;
   
     arma::mat dictionary;
   
     double lambda;
   
     size_t maxIterations;
     double tolerance;
   };
   
   } // namespace lcc
   } // namespace mlpack
   
   // Include implementation.
   #include "lcc_impl.hpp"
   
   #endif
