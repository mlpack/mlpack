
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_decomposition_policies_svdplusplus_method.hpp:

Program Listing for File svdplusplus_method.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_decomposition_policies_svdplusplus_method.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/decomposition_policies/svdplusplus_method.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_SVDPLUSPLUS_METHOD_HPP
   #define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_SVDPLUSPLUS_METHOD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/svdplusplus/svdplusplus.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class SVDPlusPlusPolicy
   {
    public:
     SVDPlusPlusPolicy(const size_t maxIterations = 10,
                       const double alpha = 0.001,
                       const double lambda = 0.1) :
         maxIterations(maxIterations),
         alpha(alpha),
         lambda(lambda)
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
       svd::SVDPlusPlus<> svdpp(maxIterations, alpha, lambda);
   
       // Save implicit data in the form of sparse matrix.
       arma::mat implicitDenseData = data.submat(0, 0, 1, data.n_cols - 1);
       svdpp.CleanData(implicitDenseData, implicitData, data);
   
       // Perform decomposition using the svdplusplus algorithm.
       svdpp.Apply(data, implicitDenseData, rank, w, h, p, q, y);
     }
   
     double GetRating(const size_t user, const size_t item) const
     {
       // Iterate through each item which the user interacted with to calculate
       // user vector.
       arma::vec userVec(h.n_rows, arma::fill::zeros);
       arma::sp_mat::const_iterator it = implicitData.begin_col(user);
       arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
       size_t implicitCount = 0;
       for (; it != it_end; ++it)
       {
         userVec += y.col(it.row());
         implicitCount += 1;
       }
       if (implicitCount != 0)
         userVec /= std::sqrt(implicitCount);
       userVec += h.col(user);
   
       double rating =
           arma::as_scalar(w.row(item) * userVec) + p(item) + q(user);
       return rating;
     }
   
     void GetRatingOfUser(const size_t user, arma::vec& rating) const
     {
       // Iterate through each item which the user interacted with to calculate
       // user vector.
       arma::vec userVec(h.n_rows, arma::fill::zeros);
       arma::sp_mat::const_iterator it = implicitData.begin_col(user);
       arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
       size_t implicitCount = 0;
       for (; it != it_end; ++it)
       {
         userVec += y.col(it.row());
         implicitCount += 1;
       }
       if (implicitCount != 0)
         userVec /= std::sqrt(implicitCount);
       userVec += h.col(user);
   
       rating = w * userVec + p + q(user);
     }
   
     template<typename NeighborSearchPolicy>
     void GetNeighborhood(const arma::Col<size_t>& users,
                          const size_t numUsersForSimilarity,
                          arma::Mat<size_t>& neighborhood,
                          arma::mat& similarities) const
     {
       // User latent vectors (matrix H) are used for neighbor search.
       // Temporarily store feature vector of queried users.
       arma::mat query(h.n_rows, users.n_elem);
       // Select feature vectors of queried users.
       for (size_t i = 0; i < users.n_elem; ++i)
         query.col(i) = h.col(users(i));
   
       NeighborSearchPolicy neighborSearch(h);
       neighborSearch.Search(
           query, numUsersForSimilarity, neighborhood, similarities);
     }
   
     const arma::mat& W() const { return w; }
     const arma::mat& H() const { return h; }
     const arma::vec& Q() const { return q; }
     const arma::vec& P() const { return p; }
     const arma::mat& Y() const { return y; }
     const arma::sp_mat& ImplicitData() const { return implicitData; }
   
     size_t MaxIterations() const { return maxIterations; }
     size_t& MaxIterations() { return maxIterations; }
   
     double Alpha() const { return alpha; }
     double& Alpha() { return alpha; }
   
     double Lambda() const { return lambda; }
     double& Lambda() { return lambda; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(maxIterations));
       ar(CEREAL_NVP(alpha));
       ar(CEREAL_NVP(lambda));
       ar(CEREAL_NVP(w));
       ar(CEREAL_NVP(h));
       ar(CEREAL_NVP(p));
       ar(CEREAL_NVP(q));
       ar(CEREAL_NVP(y));
       ar(CEREAL_NVP(implicitData));
     }
   
    private:
     size_t maxIterations;
     double alpha;
     double lambda;
     arma::mat w;
     arma::mat h;
     arma::vec p;
     arma::vec q;
     arma::mat y;
     arma::sp_mat implicitData;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
