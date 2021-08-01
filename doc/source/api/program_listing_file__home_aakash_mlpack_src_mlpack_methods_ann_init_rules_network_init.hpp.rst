
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_network_init.hpp:

Program Listing for File network_init.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_init_rules_network_init.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/init_rules/network_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_INIT_RULES_NETWORK_INIT_HPP
   #define MLPACK_METHODS_ANN_INIT_RULES_NETWORK_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "../visitor/reset_visitor.hpp"
   #include "../visitor/weight_size_visitor.hpp"
   #include "../visitor/weight_set_visitor.hpp"
   #include "init_rules_traits.hpp"
   
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InitializationRuleType, typename... CustomLayers>
   class NetworkInitialization
   {
    public:
     NetworkInitialization(
         const InitializationRuleType& initializeRule = InitializationRuleType()) :
         initializeRule(initializeRule)
     {
       // Nothing to do here.
     }
   
     template <typename eT>
     void Initialize(const std::vector<LayerTypes<CustomLayers...> >& network,
                     arma::Mat<eT>& parameter, size_t parameterOffset = 0)
     {
       // Determine the number of parameter/weights of the given network.
       if (parameter.is_empty())
       {
         size_t weights = 0;
         for (size_t i = 0; i < network.size(); ++i)
           weights += boost::apply_visitor(weightSizeVisitor, network[i]);
         parameter.set_size(weights, 1);
       }
   
       // Initialize the network layer by layer or the complete network.
       if (ann::InitTraits<InitializationRuleType>::UseLayer)
       {
         for (size_t i = 0, offset = parameterOffset; i < network.size(); ++i)
         {
           // Initialize the layer with the specified parameter/weight
           // initialization rule.
           const size_t weight = boost::apply_visitor(weightSizeVisitor,
               network[i]);
           arma::Mat<eT> tmp = arma::mat(parameter.memptr() + offset,
               weight, 1, false, false);
           initializeRule.Initialize(tmp, tmp.n_elem, 1);
   
           // Increase the parameter/weight offset for the next layer.
           offset += weight;
         }
       }
       else
       {
         initializeRule.Initialize(parameter, parameter.n_elem, 1);
       }
   
       // Note: We can't merge the for loop into the for loop above because
       // WeightSetVisitor also sets the parameter/weights of the inner modules.
       // Inner Modules are held by the parent module e.g. the concat module can
       // hold various other modules.
       for (size_t i = 0, offset = parameterOffset; i < network.size(); ++i)
       {
         offset += boost::apply_visitor(WeightSetVisitor(parameter, offset),
             network[i]);
   
         boost::apply_visitor(resetVisitor, network[i]);
       }
     }
   
    private:
     InitializationRuleType initializeRule;
   
     ResetVisitor resetVisitor;
   
     WeightSizeVisitor weightSizeVisitor;
   }; // class NetworkInitialization
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
