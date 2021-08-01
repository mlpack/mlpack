
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_sparse_coding_impl.hpp:

Program Listing for File sparse_coding_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_coding_sparse_coding_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_coding/sparse_coding_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_IMPL_HPP
   #define MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_IMPL_HPP
   
   // In case it hasn't already been included.
   #include "sparse_coding.hpp"
   
   namespace mlpack {
   namespace sparse_coding {
   
   template<typename DictionaryInitializer>
   SparseCoding::SparseCoding(
       const arma::mat& data,
       const size_t atoms,
       const double lambda1,
       const double lambda2,
       const size_t maxIterations,
       const double objTolerance,
       const double newtonTolerance,
       const DictionaryInitializer& initializer) :
       atoms(atoms),
       lambda1(lambda1),
       lambda2(lambda2),
       maxIterations(maxIterations),
       objTolerance(objTolerance),
       newtonTolerance(newtonTolerance)
   {
     Train(data, initializer);
   }
   
   template<typename DictionaryInitializer>
   double SparseCoding::Train(
       const arma::mat& data,
       const DictionaryInitializer& initializer)
   {
     // Now, train.
     Timer::Start("sparse_coding");
   
     // Initialize the dictionary.
     initializer.Initialize(data, atoms, dictionary);
   
     double lastObjVal = DBL_MAX;
   
     // Take the initial coding step, which has to happen before entering the main
     // optimization loop.
     Log::Info << "Initial coding step." << std::endl;
   
     arma::mat codes(atoms, data.n_cols);
     Encode(data, codes);
     arma::uvec adjacencies = find(codes);
   
     Log::Info << "  Sparsity level: " << 100.0 * ((double) (adjacencies.n_elem))
         / ((double) (atoms * data.n_cols)) << "%." << std::endl;
     Log::Info << "  Objective value: " << Objective(data, codes) << "."
         << std::endl;
   
     for (size_t t = 1; t != maxIterations; ++t)
     {
       // Print current iteration, and maximum number of iterations (if it isn't
       // 0).
       Log::Info << "Iteration " << t;
       if (maxIterations != 0)
         Log::Info << " of " << maxIterations;
       Log::Info << "." << std::endl;
   
       // First step: optimize the dictionary.
       Log::Info << "Performing dictionary step... " << std::endl;
       OptimizeDictionary(data, codes, adjacencies);
       Log::Info << "  Objective value: " << Objective(data, codes) << "."
           << std::endl;
   
       // Second step: perform the coding.
       Log::Info << "Performing coding step..." << std::endl;
       Encode(data, codes);
       // Get the indices of all the nonzero elements in the codes.
       adjacencies = find(codes);
       Log::Info << "  Sparsity level: " << 100.0 * ((double) (adjacencies.n_elem))
           / ((double) (atoms * data.n_cols)) << "%." << std::endl;
   
       // Find the new objective value and improvement so we can check for
       // convergence.
       double curObjVal = Objective(data, codes);
       double improvement = lastObjVal - curObjVal;
       Log::Info << "  Objective value: " << curObjVal << " (improvement "
           << std::scientific << improvement << ")." << std::endl;
   
       lastObjVal = curObjVal;
   
       // Have we converged?
       if (improvement < objTolerance)
       {
         Log::Info << "Converged within tolerance " << objTolerance << ".\n";
         break;
       }
     }
   
     Timer::Stop("sparse_coding");
     return lastObjVal;
   }
   
   template<typename Archive>
   void SparseCoding::serialize(Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(atoms));
     ar(CEREAL_NVP(dictionary));
     ar(CEREAL_NVP(lambda1));
     ar(CEREAL_NVP(lambda2));
     ar(CEREAL_NVP(maxIterations));
     ar(CEREAL_NVP(objTolerance));
     ar(CEREAL_NVP(newtonTolerance));
   }
   
   } // namespace sparse_coding
   } // namespace mlpack
   
   #endif
