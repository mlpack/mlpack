
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_run_set_visitor_impl.hpp:

Program Listing for File run_set_visitor_impl.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_run_set_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/run_set_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "run_set_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline RunSetVisitor::RunSetVisitor(
       const bool run) : run(run)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void RunSetVisitor::operator()(LayerType* layer) const
   {
     LayerRun(layer);
   }
   
   inline void RunSetVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasRunCheck<T, bool&(T::*)(void)>::value &&
       HasModelCheck<T>::value, void>::type
   RunSetVisitor::LayerRun(T* layer) const
   {
     layer->Run() = run;
   
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(RunSetVisitor(run),
           layer->Model()[i]);
     }
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasRunCheck<T, bool&(T::*)(void)>::value &&
       HasModelCheck<T>::value, void>::type
   RunSetVisitor::LayerRun(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(RunSetVisitor(run),
           layer->Model()[i]);
     }
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasRunCheck<T, bool&(T::*)(void)>::value &&
       !HasModelCheck<T>::value, void>::type
   RunSetVisitor::LayerRun(T* layer) const
   {
     layer->Run() = run;
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasRunCheck<T, bool&(T::*)(void)>::value &&
       !HasModelCheck<T>::value, void>::type
   RunSetVisitor::LayerRun(T* /* input */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
