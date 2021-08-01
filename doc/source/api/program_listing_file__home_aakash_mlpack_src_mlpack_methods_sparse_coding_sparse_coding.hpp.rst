
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_sparse_coding.hpp:

Program Listing for File sparse_coding.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_sparse_coding.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_coding/sparse_coding.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP
   #define MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/lars/lars.hpp>
   
   // Include our three simple dictionary initializers.
   #include "nothing_initializer.hpp"
   #include "data_dependent_random_initializer.hpp"
   #include "random_initializer.hpp"
   
   namespace mlpack {
   namespace sparse_coding {
   
   class SparseCoding
   {
    public:
     template<typename DictionaryInitializer = DataDependentRandomInitializer>
     SparseCoding(const arma::mat& data,
                  const size_t atoms,
                  const double lambda1,
                  const double lambda2 = 0,
                  const size_t maxIterations = 0,
                  const double objTolerance = 0.01,
                  const double newtonTolerance = 1e-6,
                  const DictionaryInitializer& initializer =
                      DictionaryInitializer());
   
     SparseCoding(const size_t atoms = 0,
                  const double lambda1 = 0,
                  const double lambda2 = 0,
                  const size_t maxIterations = 0,
                  const double objTolerance = 0.01,
                  const double newtonTolerance = 1e-6);
   
     template<typename DictionaryInitializer = DataDependentRandomInitializer>
     double Train(const arma::mat& data,
                  const DictionaryInitializer& initializer =
                      DictionaryInitializer());
   
     void Encode(const arma::mat& data, arma::mat& codes);
   
     double OptimizeDictionary(const arma::mat& data,
                               const arma::mat& codes,
                               const arma::uvec& adjacencies);
   
     void ProjectDictionary();
   
     double Objective(const arma::mat& data, const arma::mat& codes) const;
   
     const arma::mat& Dictionary() const { return dictionary; }
     arma::mat& Dictionary() { return dictionary; }
   
     size_t Atoms() const { return atoms; }
     size_t& Atoms() { return atoms; }
   
     double Lambda1() const { return lambda1; }
     double& Lambda1() { return lambda1; }
   
     double Lambda2() const { return lambda2; }
     double& Lambda2() { return lambda2; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     double ObjTolerance() const { return objTolerance; }
     double& ObjTolerance() { return objTolerance; }
   
     double NewtonTolerance() const { return newtonTolerance; }
     double& NewtonTolerance() { return newtonTolerance; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t atoms;
   
     arma::mat dictionary;
   
     double lambda1;
     double lambda2;
   
     size_t maxIterations;
     double objTolerance;
     double newtonTolerance;
   };
   
   } // namespace sparse_coding
   } // namespace mlpack
   
   // Include implementation.
   #include "sparse_coding_impl.hpp"
   
   #endif
