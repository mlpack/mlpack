
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_orthogonal_init.hpp:

Program Listing for File orthogonal_init.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_orthogonal_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/orthogonal_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class OrthogonalInitialization
   {
    public:
     OrthogonalInitialization(const double gain = 1.0) : gain(gain) { }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
     {
       arma::Mat<eT> V;
       arma::Col<eT> s;
   
       arma::svd_econ(W, s, V, arma::randu<arma::Mat<eT> >(rows, cols));
       W *= gain;
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       arma::Mat<eT> V;
       arma::Col<eT> s;
   
       arma::svd_econ(W, s, V, arma::randu<arma::Mat<eT> >(W.n_rows, W.n_cols));
       W *= gain;
     }
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W,
                     const size_t rows,
                     const size_t cols,
                     const size_t slices)
     {
       if (W.is_empty())
         W.set_size(rows, cols, slices);
   
       for (size_t i = 0; i < slices; ++i)
         Initialize(W.slice(i), rows, cols);
     }
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty cube." << std::endl;
   
       for (size_t i = 0; i < W.n_slices; ++i)
         Initialize(W.slice(i));
     }
   
    private:
     double gain;
   }; // class OrthogonalInitialization
   
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
