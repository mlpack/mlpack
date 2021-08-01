
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_backward_visitor.hpp:

Program Listing for File backward_visitor.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_backward_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/backward_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class BackwardVisitor : public boost::static_visitor<void>
   {
    public:
     BackwardVisitor(const arma::mat& input,
                     const arma::mat& error,
                     arma::mat& delta);
   
     BackwardVisitor(const arma::mat& input,
                     const arma::mat& error,
                     arma::mat& delta,
                     const size_t index);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     const arma::mat& input;
   
     const arma::mat& error;
   
     arma::mat& delta;
   
     size_t index;
   
     bool hasIndex;
   
     template<typename T>
     typename std::enable_if<
         !HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
     LayerBackward(T* layer, arma::mat& input) const;
   
     template<typename T>
     typename std::enable_if<
         HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
     LayerBackward(T* layer, arma::mat& input) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "backward_visitor_impl.hpp"
   
   #endif
