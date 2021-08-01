
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_lsh_lsh_search.hpp:

Program Listing for File lsh_search.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_lsh_lsh_search.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/lsh/lsh_search.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

    title={Locality-sensitive hashing scheme based on p-stable distributions},
    author={Datar, M. and Immorlica, N. and Indyk, P. and Mirrokni, V.S.},
    booktitle=
        {Proceedings of the 12th Annual Symposium on Computational Geometry},
    pages={253--262},
    year={2004},
    organization={ACM}
   }
   
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
   
   #include <queue>
   
   namespace mlpack {
   namespace neighbor {
   
   template<
       typename SortPolicy = NearestNeighborSort,
       typename MatType = arma::mat
   >
   class LSHSearch
   {
    public:
     LSHSearch(MatType referenceSet,
               const arma::cube& projections,
               const double hashWidth = 0.0,
               const size_t secondHashSize = 99901,
               const size_t bucketSize = 500);
   
     LSHSearch(MatType referenceSet,
               const size_t numProj,
               const size_t numTables,
               const double hashWidth = 0.0,
               const size_t secondHashSize = 99901,
               const size_t bucketSize = 500);
   
     LSHSearch();
   
     LSHSearch(const LSHSearch& other);
   
     LSHSearch(LSHSearch&& other);
   
     LSHSearch& operator=(const LSHSearch& other);
   
     LSHSearch& operator=(LSHSearch&& other);
   
     void Train(MatType referenceSet,
                const size_t numProj,
                const size_t numTables,
                const double hashWidth = 0.0,
                const size_t secondHashSize = 99901,
                const size_t bucketSize = 500,
                const arma::cube& projection = arma::cube());
   
     void Search(const MatType& querySet,
                 const size_t k,
                 arma::Mat<size_t>& resultingNeighbors,
                 arma::mat& distances,
                 const size_t numTablesToSearch = 0,
                 const size_t T = 0);
   
     void Search(const size_t k,
                 arma::Mat<size_t>& resultingNeighbors,
                 arma::mat& distances,
                 const size_t numTablesToSearch = 0,
                 size_t T = 0);
   
     static double ComputeRecall(const arma::Mat<size_t>& foundNeighbors,
                                 const arma::Mat<size_t>& realNeighbors);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
     size_t DistanceEvaluations() const { return distanceEvaluations; }
     size_t& DistanceEvaluations() { return distanceEvaluations; }
   
     const MatType& ReferenceSet() const { return referenceSet; }
   
     size_t NumProjections() const { return projections.n_slices; }
   
     const arma::mat& Offsets() const { return offsets; }
   
     const arma::vec& SecondHashWeights() const { return secondHashWeights; }
   
     size_t BucketSize() const { return bucketSize; }
   
     const std::vector<arma::Col<size_t>>& SecondHashTable() const
         { return secondHashTable; }
   
     const arma::cube& Projections() { return projections; }
   
     void Projections(const arma::cube& projTables)
     {
       // Simply call Train() with the given projection tables.
       Train(referenceSet, numProj, numTables, hashWidth, secondHashSize,
           bucketSize, projTables);
     }
   
    private:
     template<typename VecType>
     void ReturnIndicesFromTable(const VecType& queryPoint,
                                 arma::uvec& referenceIndices,
                                 size_t numTablesToSearch,
                                 const size_t T) const;
   
     void BaseCase(const size_t queryIndex,
                   const arma::uvec& referenceIndices,
                   const size_t k,
                   arma::Mat<size_t>& neighbors,
                   arma::mat& distances) const;
   
     void BaseCase(const size_t queryIndex,
                   const arma::uvec& referenceIndices,
                   const size_t k,
                   const MatType& querySet,
                   arma::Mat<size_t>& neighbors,
                   arma::mat& distances) const;
   
     void GetAdditionalProbingBins(const arma::vec& queryCode,
                                   const arma::vec& queryCodeNotFloored,
                                   const size_t T,
                                   arma::mat& additionalProbingBins) const;
   
     double PerturbationScore(const std::vector<bool>& A,
                              const arma::vec& scores) const;
   
     bool PerturbationShift(std::vector<bool>& A) const;
   
     bool PerturbationExpand(std::vector<bool>& A) const;
   
     bool PerturbationValid(const std::vector<bool>& A) const;
   
     MatType referenceSet;
   
     size_t numProj;
     size_t numTables;
   
     arma::cube projections; // should be [numProj x dims] x numTables slices
   
     arma::mat offsets; // should be numProj x numTables
   
     double hashWidth;
   
     size_t secondHashSize;
   
     arma::vec secondHashWeights;
   
     size_t bucketSize;
   
     std::vector<arma::Col<size_t>> secondHashTable;
   
     arma::Col<size_t> bucketContentSize;
   
     arma::Col<size_t> bucketRowInHashTable;
   
     size_t distanceEvaluations;
   
     typedef std::pair<double, size_t> Candidate;
   
     struct CandidateCmp {
       bool operator()(const Candidate& c1, const Candidate& c2)
       {
         return !SortPolicy::IsBetter(c2.first, c1.first);
       };
     };
   
     typedef std::priority_queue<Candidate, std::vector<Candidate>, CandidateCmp>
         CandidateList;
   }; // class LSHSearch
   
   } // namespace neighbor
   } // namespace mlpack
   
   // Include implementation.
   #include "lsh_search_impl.hpp"
   
   #endif
