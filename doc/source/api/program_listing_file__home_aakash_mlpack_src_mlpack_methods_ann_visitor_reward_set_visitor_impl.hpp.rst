
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reward_set_visitor_impl.hpp:

Program Listing for File reward_set_visitor_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_visitor_reward_set_visitor_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/visitor/reward_set_visitor_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_IMPL_HPP
   #define MLPACK_METHODS_ANN_VISITOR_REWARD_SET_VISITOR_IMPL_HPP
   
   // In case it hasn't been included yet.
   #include "reward_set_visitor.hpp"
   
   namespace mlpack {
   namespace ann {
   
   inline RewardSetVisitor::RewardSetVisitor(const double reward) : reward(reward)
   {
     /* Nothing to do here. */
   }
   
   template<typename LayerType>
   inline void RewardSetVisitor::operator()(LayerType* layer) const
   {
     LayerReward(layer);
   }
   
   inline void RewardSetVisitor::operator()(MoreTypes layer) const
   {
     layer.apply_visitor(*this);
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasRewardCheck<T, double&(T::*)()>::value &&
       HasModelCheck<T>::value, void>::type
   RewardSetVisitor::LayerReward(T* layer) const
   {
     layer->Reward() = reward;
   
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(RewardSetVisitor(reward),
           layer->Model()[i]);
     }
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasRewardCheck<T, double&(T::*)()>::value &&
       HasModelCheck<T>::value, void>::type
   RewardSetVisitor::LayerReward(T* layer) const
   {
     for (size_t i = 0; i < layer->Model().size(); ++i)
     {
       boost::apply_visitor(RewardSetVisitor(reward),
           layer->Model()[i]);
     }
   }
   
   template<typename T>
   inline typename std::enable_if<
       HasRewardCheck<T, double&(T::*)()>::value &&
       !HasModelCheck<T>::value, void>::type
   RewardSetVisitor::LayerReward(T* layer) const
   {
     layer->Reward() = reward;
   }
   
   template<typename T>
   inline typename std::enable_if<
       !HasRewardCheck<T, double&(T::*)()>::value &&
       !HasModelCheck<T>::value, void>::type
   RewardSetVisitor::LayerReward(T* /* input */) const
   {
     /* Nothing to do here. */
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
