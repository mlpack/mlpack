
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_vr_class_reward.hpp:

Program Listing for File vr_class_reward.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_vr_class_reward.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/vr_class_reward.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_HPP
   #define MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "layer_types.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class VRClassReward
   {
    public:
     VRClassReward(const double scale = 1, const bool sizeAverage = true);
   
     template<typename InputType, typename TargetType>
     double Forward(const InputType& input, const TargetType& target);
   
     template<typename InputType, typename TargetType, typename OutputType>
     void Backward(const InputType& input,
                   const TargetType& target,
                   OutputType& output);
   
     OutputDataType& OutputParameter() const {return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType& Delta() const {return delta; }
     OutputDataType& Delta() { return delta; }
   
     /*
      * Add a new module to the model.
      *
      * @param args The layer parameter.
      */
     template <class LayerType, class... Args>
     void Add(Args... args) { network.push_back(new LayerType(args...)); }
   
     /*
      * Add a new module to the model.
      *
      * @param layer The Layer to be added to the model.
      */
     void Add(LayerTypes<> layer) { network.push_back(layer); }
   
     std::vector<LayerTypes<> >& Model() { return network; }
   
     bool SizeAverage() const { return sizeAverage; }
   
     double Scale() const { return scale; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */);
   
    private:
     double scale;
   
     bool sizeAverage;
   
     double reward;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   
     std::vector<LayerTypes<> > network;
   }; // class VRClassReward
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "vr_class_reward_impl.hpp"
   
   #endif
