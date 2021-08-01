
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_gru.hpp:

Program Listing for File gru.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_gru.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/gru.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

      title     = {Gated Feedback Recurrent Neural Networks.},
      author    = {Chung, Junyoung and G{\"u}l{\c{c}}ehre, Caglar and Cho,
                  Kyunghyun and Bengio, Yoshua},
      booktitle = {ICML},
      pages     = {2067--2075},
      year      = {2015},
      url       = {https://arxiv.org/abs/1502.02367}
   }
   
   #ifndef MLPACK_METHODS_ANN_LAYER_GRU_HPP
   #define MLPACK_METHODS_ANN_LAYER_GRU_HPP
   
   #include <list>
   #include <limits>
   
   #include <mlpack/prereqs.hpp>
   
   #include "../visitor/delta_visitor.hpp"
   #include "../visitor/output_parameter_visitor.hpp"
   
   #include "layer_types.hpp"
   #include "add_merge.hpp"
   #include "sequential.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class GRU
   {
    public:
     GRU();
   
     GRU(const size_t inSize,
         const size_t outSize,
         const size_t rho = std::numeric_limits<size_t>::max());
   
     template<typename eT>
     void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g);
   
     /*
      * Calculate the gradient using the output delta and the input activation.
      *
      * @param input The input parameter used for calculating the gradient.
      * @param error The calculated error.
      * @param gradient The calculated gradient.
      */
     template<typename eT>
     void Gradient(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& /* error */,
                   arma::Mat<eT>& /* gradient */);
   
     /*
      * Resets the cell to accept a new input. This breaks the BPTT chain starts a
      * new one.
      *
      * @param size The current maximum number of steps through time.
      */
     void ResetCell(const size_t size);
   
     bool Deterministic() const { return deterministic; }
     bool& Deterministic() { return deterministic; }
   
     size_t Rho() const { return rho; }
     size_t& Rho() { return rho; }
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return gradient; }
     OutputDataType& Gradient() { return gradient; }
   
     std::vector<LayerTypes<> >& Model() { return network; }
   
     size_t InSize() const { return inSize; }
   
     size_t OutSize() const { return outSize; }
   
     size_t InputShape() const
     {
       return inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inSize;
   
     size_t outSize;
   
     size_t rho;
   
     size_t batchSize;
   
     OutputDataType weights;
   
     LayerTypes<> input2GateModule;
   
     LayerTypes<> output2GateModule;
   
     LayerTypes<> outputHidden2GateModule;
   
     LayerTypes<> inputGateModule;
   
     LayerTypes<> hiddenStateModule;
   
     LayerTypes<> forgetGateModule;
   
     OutputParameterVisitor outputParameterVisitor;
   
     DeltaVisitor deltaVisitor;
   
     DeleteVisitor deleteVisitor;
   
     std::vector<LayerTypes<> > network;
   
     size_t forwardStep;
   
     size_t backwardStep;
   
     size_t gradientStep;
   
     std::list<arma::mat> outParameter;
   
     arma::mat allZeros;
   
     std::list<arma::mat>::iterator prevOutput;
   
     std::list<arma::mat>::iterator backIterator;
   
     std::list<arma::mat>::iterator gradIterator;
   
     arma::mat prevError;
   
     bool deterministic;
   
     OutputDataType delta;
   
     OutputDataType gradient;
   
     OutputDataType outputParameter;
   }; // class GRU
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "gru_impl.hpp"
   
   #endif
