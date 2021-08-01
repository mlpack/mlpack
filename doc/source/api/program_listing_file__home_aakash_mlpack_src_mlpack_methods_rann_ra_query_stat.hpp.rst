
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_rann_ra_query_stat.hpp:

Program Listing for File ra_query_stat.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_rann_ra_query_stat.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/rann/ra_query_stat.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANN_RA_QUERY_STAT_HPP
   #define MLPACK_METHODS_RANN_RA_QUERY_STAT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include <mlpack/core/tree/binary_space_tree.hpp>
   
   #include <mlpack/core/metrics/lmetric.hpp>
   #include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename SortPolicy>
   class RAQueryStat
   {
    public:
     RAQueryStat() : bound(SortPolicy::WorstDistance()), numSamplesMade(0) { }
   
     template<typename TreeType>
     RAQueryStat(const TreeType& /* node */) :
       bound(SortPolicy::WorstDistance()),
       numSamplesMade(0)
     { }
   
     double Bound() const { return bound; }
     double& Bound() { return bound; }
   
     size_t NumSamplesMade() const { return numSamplesMade; }
     size_t& NumSamplesMade() { return numSamplesMade; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(bound));
       ar(CEREAL_NVP(numSamplesMade));
     }
   
    private:
     double bound;
     size_t numSamplesMade;
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
