
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_interpolation_policies_regression_interpolation.hpp:

Program Listing for File regression_interpolation.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_interpolation_policies_regression_interpolation.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/interpolation_policies/regression_interpolation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_REGRESSION_INTERPOLATION_HPP
   #define MLPACK_METHODS_CF_REGRESSION_INTERPOLATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class RegressionInterpolation
   {
    public:
     RegressionInterpolation() { }
   
     RegressionInterpolation(const arma::sp_mat& cleanedData)
     {
       const size_t userNum = cleanedData.n_cols;
       a.set_size(userNum, userNum);
       b.set_size(userNum, userNum);
     }
   
     template <typename VectorType,
               typename DecompositionPolicy>
     void GetWeights(VectorType&& weights,
                     const DecompositionPolicy& decomposition,
                     const size_t queryUser,
                     const arma::Col<size_t>& neighbors,
                     const arma::vec& /* similarities*/,
                     const arma::sp_mat& cleanedData)
     {
       if (weights.n_elem != neighbors.n_elem)
       {
         Log::Fatal << "The size of the first parameter (weights) should "
             << "be set to the number of neighbors before calling GetWeights()."
             << std::endl;
       }
   
       const arma::mat& w = decomposition.W();
       const arma::mat& h = decomposition.H();
       const size_t itemNum = cleanedData.n_rows;
       const size_t neighborNum = neighbors.size();
   
       // Coeffcients of the linear equations used to compute weights.
       arma::mat coeff(neighborNum, neighborNum);
       // Constant terms of the linear equations used to compute weights.
       arma::vec constant(neighborNum);
   
       arma::vec userRating(cleanedData.col(queryUser));
       const size_t support = arma::accu(userRating != 0);
   
       // If user has no rating at all, average interpolation is used.
       if (support == 0)
       {
         weights.fill(1.0 / neighbors.n_elem);
         return;
       }
   
       for (size_t i = 0; i < neighborNum; ++i)
       {
         // Calculate coefficient.
         arma::vec iPrediction;
         for (size_t j = i; j < neighborNum; ++j)
         {
           if (a(neighbors(i), neighbors(j)) != 0)
           {
             // The coefficient has already been cached.
             coeff(i, j) = a(neighbors(i), neighbors(j));
             coeff(j, i) = coeff(i, j);
           }
           else
           {
             // Calculate the coefficient.
             if (iPrediction.size() == 0)
               // Avoid recalculation of iPrediction.
               iPrediction = w * h.col(neighbors(i));
             arma::vec jPrediction = w * h.col(neighbors(j));
             coeff(i, j) = arma::dot(iPrediction, jPrediction) / itemNum;
             if (coeff(i, j) == 0)
               coeff(i, j) = std::numeric_limits<double>::min();
             coeff(j, i) = coeff(i, j);
             // Cache calcualted coefficient.
             a(neighbors(i), neighbors(j)) = coeff(i, j);
             a(neighbors(j), neighbors(i)) = coeff(i, j);
           }
         }
   
         // Calculate constant terms.
         if (b(neighbors(i), queryUser) != 0)
           // The constant term has already been cached.
           constant(i) = b(neighbors(i), queryUser);
         else
         {
           // Calcuate the constant term.
           if (iPrediction.size() == 0)
               // Avoid recalculation of iPrediction.
               iPrediction = w * h.col(neighbors(i));
           constant(i) = arma::dot(iPrediction, userRating) / support;
           if (constant(i) == 0)
             constant(i) = std::numeric_limits<double>::min();
           // Cache calculated constant term.
           b(neighbors(i), queryUser) = constant(i);
         }
       }
       weights = arma::solve(coeff, constant);
     }
   
    private:
     arma::sp_mat a;
     arma::sp_mat b;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
