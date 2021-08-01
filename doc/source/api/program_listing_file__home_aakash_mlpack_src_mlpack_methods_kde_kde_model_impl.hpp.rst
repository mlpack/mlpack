
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kde_kde_model_impl.hpp:

Program Listing for File kde_model_impl.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kde_kde_model_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kde/kde_model_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KDE_MODEL_IMPL_HPP
   #define MLPACK_METHODS_KDE_MODEL_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "kde_model.hpp"
   
   namespace mlpack {
   namespace kde {
   
   template<typename KernelType,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   void KDEWrapper<KernelType, TreeType>::Train(arma::mat&& referenceSet)
   {
     kde.Train(std::move(referenceSet));
   }
   
   template<typename KernelType,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   void KDEWrapper<KernelType, TreeType>::Evaluate(arma::mat&& querySet,
                                                   arma::vec& estimates)
   {
     const size_t dimension = querySet.n_rows;
     kde.Evaluate(std::move(querySet), estimates);
     KernelNormalizer::ApplyNormalizer<KernelType>(kde.Kernel(),
                                                   dimension,
                                                   estimates);
   }
   
   template<typename KernelType,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   void KDEWrapper<KernelType, TreeType>::Evaluate(arma::vec& estimates)
   {
     kde.Evaluate(estimates);
     const size_t dimension = kde.ReferenceTree()->Dataset().n_rows;
     KernelNormalizer::ApplyNormalizer<KernelType>(kde.Kernel(),
                                                   dimension,
                                                   estimates);
   }
   
   template<template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType,
            typename Archive>
   void SerializationHelper(Archive& ar,
                            KDEWrapperBase* kdeModel,
                            const KDEModel::KernelTypes kernelType)
   {
     switch (kernelType)
     {
       case KDEModel::GAUSSIAN_KERNEL:
         {
           KDEWrapper<kernel::GaussianKernel, TreeType>& typedModel =
               dynamic_cast<KDEWrapper<kernel::GaussianKernel,
                                       TreeType>&>(*kdeModel);
           ar(CEREAL_NVP(typedModel));
           break;
         }
       case KDEModel::EPANECHNIKOV_KERNEL:
         {
           KDEWrapper<kernel::EpanechnikovKernel, TreeType>& typedModel =
               dynamic_cast<KDEWrapper<kernel::EpanechnikovKernel,
                                       TreeType>&>(*kdeModel);
           ar(CEREAL_NVP(typedModel));
           break;
         }
       case KDEModel::LAPLACIAN_KERNEL:
         {
           KDEWrapper<kernel::LaplacianKernel, TreeType>& typedModel =
               dynamic_cast<KDEWrapper<kernel::LaplacianKernel,
                                       TreeType>&>(*kdeModel);
           ar(CEREAL_NVP(typedModel));
           break;
         }
       case KDEModel::SPHERICAL_KERNEL:
         {
           KDEWrapper<kernel::SphericalKernel, TreeType>& typedModel =
               dynamic_cast<KDEWrapper<kernel::SphericalKernel,
                                       TreeType>&>(*kdeModel);
           ar(CEREAL_NVP(typedModel));
           break;
         }
       case KDEModel::TRIANGULAR_KERNEL:
         {
           KDEWrapper<kernel::TriangularKernel, TreeType>& typedModel =
               dynamic_cast<KDEWrapper<kernel::TriangularKernel,
                                       TreeType>&>(*kdeModel);
           ar(CEREAL_NVP(typedModel));
           break;
         }
     }
   }
   
   // Serialize the model.
   template<typename Archive>
   void KDEModel::serialize(Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(bandwidth));
     ar(CEREAL_NVP(relError));
     ar(CEREAL_NVP(absError));
     ar(CEREAL_NVP(kernelType));
     ar(CEREAL_NVP(treeType));
     ar(CEREAL_NVP(monteCarlo));
     ar(CEREAL_NVP(mcProb));
     ar(CEREAL_NVP(initialSampleSize));
     ar(CEREAL_NVP(mcEntryCoef));
     ar(CEREAL_NVP(mcBreakCoef));
   
     if (cereal::is_loading<Archive>())
     {
       monteCarlo = KDEDefaultParams::monteCarlo;
       mcProb = KDEDefaultParams::mcProb;
       initialSampleSize = KDEDefaultParams::initialSampleSize;
       mcEntryCoef = KDEDefaultParams::mcEntryCoef;
       mcBreakCoef = KDEDefaultParams::mcBreakCoef;
     }
   
     if (cereal::is_loading<Archive>())
       InitializeModel(); // Values will be overwritten.
   
     // Avoid polymorphism in serialization by serializing directly by the type.
     switch (treeType)
     {
       case KD_TREE:
         SerializationHelper<tree::KDTree>(ar, kdeModel, kernelType);
         break;
   
       case BALL_TREE:
         SerializationHelper<tree::BallTree>(ar, kdeModel, kernelType);
         break;
   
       case COVER_TREE:
         SerializationHelper<tree::StandardCoverTree>(ar, kdeModel, kernelType);
         break;
   
       case OCTREE:
         SerializationHelper<tree::Octree>(ar, kdeModel, kernelType);
         break;
   
       case R_TREE:
         SerializationHelper<tree::RTree>(ar, kdeModel, kernelType);
         break;
     }
   }
   
   } // namespace kde
   } // namespace mlpack
   
   #endif
