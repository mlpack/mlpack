
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_decomposition_policies_regularized_svd_method.hpp:

Program Listing for File regularized_svd_method.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_decomposition_policies_regularized_svd_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/decomposition_policies/regularized_svd_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_REGULARIZED_SVD_METHOD_HPP
   #define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_REGULARIZED_SVD_METHOD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/regularized_svd/regularized_svd.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class RegSVDPolicy
   {
    public:
     RegSVDPolicy(const size_t maxIterations = 10) :
         maxIterations(maxIterations)
     {
       /* Nothing to do here */
     }
   
     void Apply(const arma::mat& data,
                const arma::sp_mat& /* cleanedData */,
                const size_t rank,
                const size_t maxIterations,
                const double /* minResidue */,
                const bool /* mit */)
     {
       // Do singular value decomposition using the regularized SVD algorithm.
       svd::RegularizedSVD<> regsvd(maxIterations);
       regsvd.Apply(data, rank, w, h);
     }
   
     double GetRating(const size_t user, const size_t item) const
     {
       double rating = arma::as_scalar(w.row(item) * h.col(user));
       return rating;
     }
   
     void GetRatingOfUser(const size_t user, arma::vec& rating) const
     {
       rating = w * h.col(user);
     }
   
     template<typename NeighborSearchPolicy>
     void GetNeighborhood(const arma::Col<size_t>& users,
                          const size_t numUsersForSimilarity,
                          arma::Mat<size_t>& neighborhood,
                          arma::mat& similarities) const
     {
       // We want to avoid calculating the full rating matrix, so we will do
       // nearest neighbor search only on the H matrix, using the observation that
       // if the rating matrix X = W*H, then d(X.col(i), X.col(j)) = d(W H.col(i),
       // W H.col(j)).  This can be seen as nearest neighbor search on the H
       // matrix with the Mahalanobis distance where M^{-1} = W^T W.  So, we'll
       // decompose M^{-1} = L L^T (the Cholesky decomposition), and then multiply
       // H by L^T. Then we can perform nearest neighbor search.
       arma::mat l = arma::chol(w.t() * w);
       arma::mat stretchedH = l * h; // Due to the Armadillo API, l is L^T.
   
       // Temporarily store feature vector of queried users.
       arma::mat query(stretchedH.n_rows, users.n_elem);
       // Select feature vectors of queried users.
       for (size_t i = 0; i < users.n_elem; ++i)
         query.col(i) = stretchedH.col(users(i));
   
       NeighborSearchPolicy neighborSearch(stretchedH);
       neighborSearch.Search(
           query, numUsersForSimilarity, neighborhood, similarities);
     }
   
     const arma::mat& W() const { return w; }
     const arma::mat& H() const { return h; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(w));
       ar(CEREAL_NVP(h));
     }
   
    private:
     size_t maxIterations;
     arma::mat w;
     arma::mat h;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
