
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_oivs_init.hpp:

Program Listing for File oivs_init.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_oivs_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/oivs_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     title={A weight value initialization method for improving learning
     performance of the backpropagation algorithm in neural networks},
     author={Shimodaira, H.},
     booktitle={Tools with Artificial Intelligence, 1994. Proceedings.,
     Sixth International Conference on},
     year={1994}
   }
   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_OIVS_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_OIVS_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
   
   #include "random_init.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<
       class ActivationFunction = LogisticFunction
   >
   class OivsInitialization
   {
    public:
     OivsInitialization(const double epsilon = 0.1,
                        const int k = 5,
                        const double gamma = 0.9) :
         k(k), gamma(gamma),
         b(std::abs(ActivationFunction::Inv(1 - epsilon) -
                    ActivationFunction::Inv(epsilon)))
     {
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
     {
       RandomInitialization randomInit(-gamma, gamma);
       randomInit.Initialize(W, rows, cols);
   
       W = (b / (k  * rows)) * arma::sqrt(W + 1);
     }
   
     template<typename eT>
     void Initialize(arma::Mat<eT>& W)
     {
       RandomInitialization randomInit(-gamma, gamma);
       randomInit.Initialize(W);
   
       W = (b / (k  * W.n_rows)) * arma::sqrt(W + 1);
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
     int k;
   
     double gamma;
   
     double b;
   }; // class OivsInitialization
   
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
