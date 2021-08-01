
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_matrix_completion_matrix_completion.hpp:

Program Listing for File matrix_completion.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_matrix_completion_matrix_completion.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/matrix_completion/matrix_completion.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP
   #define MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP
   
   #include <ensmallen.hpp>
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace matrix_completion {
   
   class MatrixCompletion
   {
    public:
     MatrixCompletion(const size_t m,
                      const size_t n,
                      const arma::umat& indices,
                      const arma::vec& values,
                      const size_t r);
   
     MatrixCompletion(const size_t m,
                      const size_t n,
                      const arma::umat& indices,
                      const arma::vec& values,
                      const arma::mat& initialPoint);
   
     MatrixCompletion(const size_t m,
                      const size_t n,
                      const arma::umat& indices,
                      const arma::vec& values);
   
     void Recover(arma::mat& recovered);
   
     const ens::LRSDP<ens::SDP<arma::sp_mat>>& Sdp() const
     {
       return sdp;
     }
     ens::LRSDP<ens::SDP<arma::sp_mat>>& Sdp() { return sdp; }
   
    private:
     size_t m;
     size_t n;
     arma::umat indices;
     arma::mat values;
   
     ens::LRSDP<ens::SDP<arma::sp_mat>> sdp;
   
     void CheckValues();
     void InitSDP();
   
     static size_t DefaultRank(const size_t m, const size_t n, const size_t p);
   };
   
   } // namespace matrix_completion
   } // namespace mlpack
   
   #endif
