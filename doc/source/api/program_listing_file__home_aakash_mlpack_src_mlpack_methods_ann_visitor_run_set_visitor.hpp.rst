
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_run_set_visitor.hpp:

Program Listing for File run_set_visitor.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_run_set_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/run_set_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class RunSetVisitor : public boost::static_visitor<void>
   {
    public:
     RunSetVisitor(const bool run = true);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     const bool run;
   
     template<typename T>
     typename std::enable_if<
         HasRunCheck<T, bool&(T::*)(void)>::value &&
         HasModelCheck<T>::value, void>::type
     LayerRun(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasRunCheck<T, bool&(T::*)(void)>::value &&
         HasModelCheck<T>::value, void>::type
     LayerRun(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasRunCheck<T, bool&(T::*)(void)>::value &&
         !HasModelCheck<T>::value, void>::type
     LayerRun(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasRunCheck<T, bool&(T::*)(void)>::value &&
         !HasModelCheck<T>::value, void>::type
     LayerRun(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "run_set_visitor_impl.hpp"
   
   #endif
