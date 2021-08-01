
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_det_dt_utils.hpp:

Program Listing for File dt_utils.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_det_dt_utils.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/det/dt_utils.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DET_DT_UTILS_HPP
   #define MLPACK_METHODS_DET_DT_UTILS_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "dtree.hpp"
   
   namespace mlpack {
   namespace det {
   
   template <typename MatType, typename TagType>
   void PrintLeafMembership(DTree<MatType, TagType>* dtree,
                            const MatType& data,
                            const arma::Mat<size_t>& labels,
                            const size_t numClasses,
                            const std::string& leafClassMembershipFile = "");
   
   template <typename MatType, typename TagType>
   void PrintVariableImportance(const DTree<MatType, TagType>* dtree,
                                const std::string viFile = "");
   
   template <typename MatType, typename TagType>
   DTree<MatType, TagType>* Trainer(MatType& dataset,
                                    const size_t folds,
                                    const bool useVolumeReg = false,
                                    const size_t maxLeafSize = 10,
                                    const size_t minLeafSize = 5,
                                    const std::string unprunedTreeOutput = "",
                                    const bool skipPruning = false);
   
   class PathCacher
   {
    public:
     enum PathFormat
     {
       FormatLR,
       FormatLR_ID,
       FormatID_LR
     };
   
     template<typename MatType>
     PathCacher(PathFormat fmt, DTree<MatType, int>* tree);
   
     template<typename MatType>
     void Enter(const DTree<MatType, int>* node,
                const DTree<MatType, int>* parent);
   
     template<typename MatType>
     void Leave(const DTree<MatType, int>* node,
                const DTree<MatType, int>* parent);
   
     inline const std::string& PathFor(int tag) const;
   
     inline int ParentOf(int tag) const;
   
     size_t NumNodes() const { return pathCache.size(); }
   
    protected:
     typedef std::list<std::pair<bool, int>>          PathType;
     typedef std::vector<std::pair<int, std::string>> PathCacheType;
   
     PathType      path;
     PathFormat    format;
     PathCacheType pathCache;
   
     inline std::string   BuildString();
   };
   
   } // namespace det
   } // namespace mlpack
   
   #include "dt_utils_impl.hpp"
   
   #endif // MLPACK_METHODS_DET_DT_UTILS_HPP
