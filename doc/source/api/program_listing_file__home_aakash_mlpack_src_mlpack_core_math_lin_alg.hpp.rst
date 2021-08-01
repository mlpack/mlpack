
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_lin_alg.hpp:

Program Listing for File lin_alg.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_lin_alg.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/lin_alg.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_LIN_ALG_HPP
   #define MLPACK_CORE_MATH_LIN_ALG_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace math {
   
   void VectorPower(arma::vec& vec, const double power);
   
   void Center(const arma::mat& x, arma::mat& xCentered);
   
   void WhitenUsingSVD(const arma::mat& x,
                       arma::mat& xWhitened,
                       arma::mat& whiteningMatrix);
   
   
   void RandVector(arma::vec& v);
   
   void Orthogonalize(const arma::mat& x, arma::mat& W);
   
   void Orthogonalize(arma::mat& x);
   
   void RemoveRows(const arma::mat& input,
                   const std::vector<size_t>& rowsToRemove,
                   arma::mat& output);
   
   void Svec(const arma::mat& input, arma::vec& output);
   
   void Svec(const arma::sp_mat& input, arma::sp_vec& output);
   
   void Smat(const arma::vec& input, arma::mat& output);
   
   inline size_t SvecIndex(size_t i, size_t j, size_t n);
   
   void SymKronId(const arma::mat& A, arma::mat& op);
   
   template <typename T>
   T Sign(const T x)
   {
       return (T(0) < x) - (x < T(0));
   }
   
   } // namespace math
   } // namespace mlpack
   
   // Partially include implementation
   #include "lin_alg_impl.hpp"
   
   #endif // MLPACK_CORE_MATH_LIN_ALG_HPP
