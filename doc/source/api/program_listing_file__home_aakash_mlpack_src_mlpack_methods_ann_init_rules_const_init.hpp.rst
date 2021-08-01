
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_const_init.hpp:

Program Listing for File const_init.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_const_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/const_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_CONST_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_CONST_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   class ConstInitialization
   {
    public:
     ConstInitialization(const double initVal = 0) : initVal(initVal)
     { /* Nothing to do here */ }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
     {
       if (W.is_empty())
         W.set_size(rows, cols);
   
       W.fill(initVal);
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty matrix." << std::endl;
   
       W.fill(initVal);
     }
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W,
                     const size_t rows,
                     const size_t cols,
                     const size_t slices)
     {
       if (W.is_empty())
         W.set_size(rows, cols, slices);
   
       W.fill(initVal);
     }
   
     template<typename eT>
     void Initialize(arma::Cube<eT>& W)
     {
       if (W.is_empty())
         Log::Fatal << "Cannot initialize an empty cube." << std::endl;
   
       W.fill(initVal);
     }
   
     double const& InitValue() const { return initVal; }
     double& initValue() { return initVal; }
   
    private:
     double initVal;
   }; // class ConstInitialization
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
