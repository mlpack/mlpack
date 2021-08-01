
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_approx_kfn_drusilla_select.hpp:

Program Listing for File drusilla_select.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_approx_kfn_drusilla_select.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/approx_kfn/drusilla_select.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title={Fast approximate furthest neighbors with data-dependent candidate
            selection},
     author={Curtin, R.R., and Gardner, A.B.},
     booktitle={Similarity Search and Applications},
     pages={221--235},
     year={2016},
     publisher={Springer}
   }
   
   #ifndef MLPACK_METHODS_APPROX_KFN_DRUSILLA_SELECT_HPP
   #define MLPACK_METHODS_APPROX_KFN_DRUSILLA_SELECT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename MatType = arma::mat>
   class DrusillaSelect
   {
    public:
     DrusillaSelect(const MatType& referenceSet,
                    const size_t l,
                    const size_t m);
   
     DrusillaSelect(const size_t l, const size_t m);
   
     void Train(const MatType& referenceSet,
                const size_t l = 0,
                const size_t m = 0);
   
     void Search(const MatType& querySet,
                 const size_t k,
                 arma::Mat<size_t>& neighbors,
                 arma::mat& distances);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     const MatType& CandidateSet() const { return candidateSet; }
     MatType& CandidateSet() { return candidateSet; }
   
     const arma::Col<size_t>& CandidateIndices() const { return candidateIndices; }
     arma::Col<size_t>& CandidateIndices() { return candidateIndices; }
   
    private:
     MatType candidateSet;
     arma::Col<size_t> candidateIndices;
   
     size_t l;
     size_t m;
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "drusilla_select_impl.hpp"
   
   #endif
