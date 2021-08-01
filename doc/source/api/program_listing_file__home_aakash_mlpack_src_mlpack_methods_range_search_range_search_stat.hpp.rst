
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_range_search_range_search_stat.hpp:

Program Listing for File range_search_stat.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_range_search_range_search_stat.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/range_search/range_search_stat.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP
   #define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace range {
   
   class RangeSearchStat
   {
    public:
     RangeSearchStat() : lastDistance(0.0) { }
   
     template<typename TreeType>
     RangeSearchStat(TreeType& /* node */) :
         lastDistance(0.0) { }
   
     double LastDistance() const { return lastDistance; }
     double& LastDistance() { return lastDistance; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(lastDistance));
     }
   
    private:
     double lastDistance;
   };
   
   } // namespace range
   } // namespace mlpack
   
   #endif
