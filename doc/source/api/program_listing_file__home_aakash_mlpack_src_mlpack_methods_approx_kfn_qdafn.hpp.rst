
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_approx_kfn_qdafn.hpp:

Program Listing for File qdafn.hpp
==================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_approx_kfn_qdafn.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/approx_kfn/qdafn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title={Approximate furthest neighbor in high dimensions},
     author={Pagh, R. and Silvestri, F. and Sivertsen, J. and Skala, M.},
     booktitle={Similarity Search and Applications},
     pages={3--14},
     year={2015},
     publisher={Springer}
   }
   
   #ifndef MLPACK_METHODS_APPROX_KFN_QDAFN_HPP
   #define MLPACK_METHODS_APPROX_KFN_QDAFN_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/dists/gaussian_distribution.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename MatType = arma::mat>
   class QDAFN
   {
    public:
     QDAFN(const size_t l, const size_t m);
   
     QDAFN(const MatType& referenceSet,
           const size_t l,
           const size_t m);
   
     void Train(const MatType& referenceSet,
                const size_t l = 0,
                const size_t m = 0);
   
     void Search(const MatType& querySet,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     size_t NumProjections() const { return candidateSet.size(); }
   
     const MatType& CandidateSet(const size_t t) const { return candidateSet[t]; }
     MatType& CandidateSet(const size_t t) { return candidateSet[t]; }
   
    private:
     size_t l;
     size_t m;
     arma::mat lines;
     arma::mat projections;
   
     arma::Mat<size_t> sIndices;
     arma::mat sValues;
   
     // Candidate sets; one element in the vector for each table.
     std::vector<MatType> candidateSet;
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "qdafn_impl.hpp"
   
   #endif
