
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_load_output_parameter_visitor.hpp:

Program Listing for File load_output_parameter_visitor.hpp
==========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_load_output_parameter_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/load_output_parameter_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_LOAD_OUTPUT_PARAMETER_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_LOAD_OUTPUT_PARAMETER_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class LoadOutputParameterVisitor : public boost::static_visitor<void>
   {
    public:
     LoadOutputParameterVisitor(std::vector<arma::mat>& parameter);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     std::vector<arma::mat>& parameter;
   
     template<typename T>
     typename std::enable_if<
         !HasModelCheck<T>::value, void>::type
     OutputParameter(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasModelCheck<T>::value, void>::type
     OutputParameter(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "load_output_parameter_visitor_impl.hpp"
   
   #endif
