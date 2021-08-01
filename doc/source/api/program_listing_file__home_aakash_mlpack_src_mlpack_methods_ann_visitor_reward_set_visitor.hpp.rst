
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reward_set_visitor.hpp:

Program Listing for File reward_set_visitor.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reward_set_visitor.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/reward_set_visitor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_HPP
   #define MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_HPP
   
   #include <mlpack/methods/ann/layer/layer_traits.hpp>
   
   #include <boost/variant.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class RewardSetVisitor : public boost::static_visitor<void>
   {
    public:
     RewardSetVisitor(const double reward);
   
     template<typename LayerType>
     void operator()(LayerType* layer) const;
   
     void operator()(MoreTypes layer) const;
   
    private:
     const double reward;
   
     template<typename T>
     typename std::enable_if<
         HasRewardCheck<T, double&(T::*)()>::value &&
         HasModelCheck<T>::value, void>::type
     LayerReward(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasRewardCheck<T, double&(T::*)()>::value &&
         HasModelCheck<T>::value, void>::type
     LayerReward(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         HasRewardCheck<T, double&(T::*)()>::value &&
         !HasModelCheck<T>::value, void>::type
     LayerReward(T* layer) const;
   
     template<typename T>
     typename std::enable_if<
         !HasRewardCheck<T, double&(T::*)()>::value &&
         !HasModelCheck<T>::value, void>::type
     LayerReward(T* layer) const;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "reward_set_visitor_impl.hpp"
   
   #endif
