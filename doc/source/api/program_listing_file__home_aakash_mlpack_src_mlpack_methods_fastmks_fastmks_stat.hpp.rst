
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks_stat.hpp:

Program Listing for File fastmks_stat.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_fastmks_fastmks_stat.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/fastmks/fastmks_stat.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_FASTMKS_FASTMKS_STAT_HPP
   #define MLPACK_METHODS_FASTMKS_FASTMKS_STAT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/tree/tree_traits.hpp>
   
   namespace mlpack {
   namespace fastmks {
   
   class FastMKSStat
   {
    public:
     FastMKSStat() :
         bound(-DBL_MAX),
         selfKernel(0.0),
         lastKernel(0.0),
         lastKernelNode(NULL)
     { }
   
     template<typename TreeType>
     FastMKSStat(const TreeType& node) :
         bound(-DBL_MAX),
         lastKernel(0.0),
         lastKernelNode(NULL)
     {
       // Do we have to calculate the centroid?
       if (tree::TreeTraits<TreeType>::FirstPointIsCentroid)
       {
         // If this type of tree has self-children, then maybe the evaluation is
         // already done.  These statistics are built bottom-up, so the child stat
         // should already be done.
         if ((tree::TreeTraits<TreeType>::HasSelfChildren) &&
             (node.NumChildren() > 0) &&
             (node.Point(0) == node.Child(0).Point(0)))
         {
           selfKernel = node.Child(0).Stat().SelfKernel();
         }
         else
         {
           selfKernel = sqrt(node.Metric().Kernel().Evaluate(
               node.Dataset().col(node.Point(0)),
               node.Dataset().col(node.Point(0))));
         }
       }
       else
       {
         // Calculate the centroid.
         arma::vec center;
         node.Center(center);
   
         selfKernel = sqrt(node.Metric().Kernel().Evaluate(center, center));
       }
     }
   
     double SelfKernel() const { return selfKernel; }
     double& SelfKernel() { return selfKernel; }
   
     double Bound() const { return bound; }
     double& Bound() { return bound; }
   
     double LastKernel() const { return lastKernel; }
     double& LastKernel() { return lastKernel; }
   
     void* LastKernelNode() const { return lastKernelNode; }
     void*& LastKernelNode() { return lastKernelNode; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(bound));
       ar(CEREAL_NVP(selfKernel));
   
       // Void out last kernel information on load.
       if (cereal::is_loading<Archive>())
       {
         lastKernel = 0.0;
         lastKernelNode = NULL;
       }
     }
   
    private:
     double bound;
   
     double selfKernel;
   
     double lastKernel;
   
     void* lastKernelNode;
   };
   
   } // namespace fastmks
   } // namespace mlpack
   
   #endif
