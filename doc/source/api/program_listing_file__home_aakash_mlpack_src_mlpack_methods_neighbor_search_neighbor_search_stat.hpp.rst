
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search_stat.hpp:

Program Listing for File neighbor_search_stat.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_neighbor_search_neighbor_search_stat.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/neighbor_search/neighbor_search_stat.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_STAT_HPP
   #define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_STAT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace neighbor {
   
   template<typename SortPolicy>
   class NeighborSearchStat
   {
    private:
     double firstBound;
     double secondBound;
     double auxBound;
     double lastDistance;
   
    public:
     NeighborSearchStat() :
         firstBound(SortPolicy::WorstDistance()),
         secondBound(SortPolicy::WorstDistance()),
         auxBound(SortPolicy::WorstDistance()),
         lastDistance(0.0) { }
   
     template<typename TreeType>
     NeighborSearchStat(TreeType& /* node */) :
         firstBound(SortPolicy::WorstDistance()),
         secondBound(SortPolicy::WorstDistance()),
         auxBound(SortPolicy::WorstDistance()),
         lastDistance(0.0) { }
   
     void Reset()
     {
       firstBound = SortPolicy::WorstDistance();
       secondBound = SortPolicy::WorstDistance();
       auxBound = SortPolicy::WorstDistance();
       lastDistance = 0.0;
     }
   
     double FirstBound() const { return firstBound; }
     double& FirstBound() { return firstBound; }
     double SecondBound() const { return secondBound; }
     double& SecondBound() { return secondBound; }
     double AuxBound() const { return auxBound; }
     double& AuxBound() { return auxBound; }
     double LastDistance() const { return lastDistance; }
     double& LastDistance() { return lastDistance; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(firstBound));
       ar(CEREAL_NVP(secondBound));
       ar(CEREAL_NVP(auxBound));
       ar(CEREAL_NVP(lastDistance));
     }
   };
   
   } // namespace neighbor
   } // namespace mlpack
   
   #endif
