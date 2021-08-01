
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_emst_dtb_stat.hpp:

Program Listing for File dtb_stat.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_emst_dtb_stat.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/emst/dtb_stat.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_EMST_DTB_STAT_HPP
   #define MLPACK_METHODS_EMST_DTB_STAT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace emst {
   
   class DTBStat
   {
    private:
     double maxNeighborDistance;
   
     double minNeighborDistance;
   
     double bound;
   
     int componentMembership;
   
    public:
     DTBStat() :
         maxNeighborDistance(DBL_MAX),
         minNeighborDistance(DBL_MAX),
         bound(DBL_MAX),
         componentMembership(-1) { }
   
     template<typename TreeType>
     DTBStat(const TreeType& node) :
         maxNeighborDistance(DBL_MAX),
         minNeighborDistance(DBL_MAX),
         bound(DBL_MAX),
         componentMembership(
             ((node.NumPoints() == 1) && (node.NumChildren() == 0)) ?
               node.Point(0) : -1) { }
   
     double MaxNeighborDistance() const { return maxNeighborDistance; }
     double& MaxNeighborDistance() { return maxNeighborDistance; }
   
     double MinNeighborDistance() const { return minNeighborDistance; }
     double& MinNeighborDistance() { return minNeighborDistance; }
   
     double Bound() const { return bound; }
     double& Bound() { return bound; }
   
     int ComponentMembership() const { return componentMembership; }
     int& ComponentMembership() { return componentMembership; }
   }; // class DTBStat
   
   } // namespace emst
   } // namespace mlpack
   
   #endif // MLPACK_METHODS_EMST_DTB_STAT_HPP
