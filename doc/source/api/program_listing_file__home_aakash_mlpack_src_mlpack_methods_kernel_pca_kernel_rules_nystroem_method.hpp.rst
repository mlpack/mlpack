
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kernel_pca_kernel_rules_nystroem_method.hpp:

Program Listing for File nystroem_method.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kernel_pca_kernel_rules_nystroem_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kernel_pca/kernel_rules/nystroem_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KERNEL_PCA_NYSTROEM_METHOD_HPP
   #define MLPACK_METHODS_KERNEL_PCA_NYSTROEM_METHOD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/nystroem_method/kmeans_selection.hpp>
   #include <mlpack/methods/nystroem_method/nystroem_method.hpp>
   
   namespace mlpack {
   namespace kpca {
   
   template<
     typename KernelType,
     typename PointSelectionPolicy = kernel::KMeansSelection<>
   >
   class NystroemKernelRule
   {
    public:
     static void ApplyKernelMatrix(const arma::mat& data,
                                   arma::mat& transformedData,
                                   arma::vec& eigval,
                                   arma::mat& eigvec,
                                   const size_t rank,
                                   KernelType kernel = KernelType())
     {
       arma::mat G, v;
       kernel::NystroemMethod<KernelType, PointSelectionPolicy> nm(data, kernel,
           rank);
       nm.Apply(G);
       transformedData = G.t() * G;
   
       // Center the reconstructed approximation.
       math::Center(transformedData, transformedData);
   
       // For PCA the data has to be centered, even if the data is centered. But
       // it is not guaranteed that the data, when mapped to the kernel space, is
       // also centered. Since we actually never work in the feature space we
       // cannot center the data. So, we perform a "psuedo-centering" using the
       // kernel matrix.
       arma::colvec colMean = arma::sum(G, 1) / G.n_rows;
       G.each_row() -= arma::sum(G, 0) / G.n_rows;
       G.each_col() -= colMean;
       G += arma::sum(colMean) / G.n_rows;
   
       // Eigendecompose the centered kernel matrix.
       transformedData = arma::symmatu(transformedData);
       if (!arma::eig_sym(eigval, eigvec, transformedData))
       {
         Log::Fatal << "Failed to construct the kernel matrix." << std::endl;
       }
   
       // Swap the eigenvalues since they are ordered backwards (we need largest
       // to smallest).
       for (size_t i = 0; i < floor(eigval.n_elem / 2.0); ++i)
         eigval.swap_rows(i, (eigval.n_elem - 1) - i);
   
       // Flip the coefficients to produce the same effect.
       eigvec = arma::fliplr(eigvec);
   
       transformedData = eigvec.t() * G.t();
     }
   };
   
   } // namespace kpca
   } // namespace mlpack
   
   #endif
