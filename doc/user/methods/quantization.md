## `Quantization`

The `Quantization` class implements various quantization techniques for compressing neural network models in mlpack. Quantization reduces the precision of model weights, typically from 32-bit floating-point to lower bit-width representations (e.g., 8-bit integers). This compression can significantly reduce model size and improve inference speed, especially on hardware optimized for lower-precision arithmetic.

Quantization is particularly useful for deploying models on resource-constrained devices or for scenarios where model size and inference speed are critical.

#### Simple usage example:

```c++
// Quantize a pre-trained neural network model to 8-bit precision:

// Load a pre-trained model (replace with your model loading code)
mlpack::FFN<> model;
mlpack::data::Load("model.bin", "model", model);

// Create a quantizer with default settings (8-bit quantization)
mlpack::Quantization quantizer;

// Apply quantization to the model
quantizer.Apply(model);

// Save the quantized model
mlpack::data::Save("quantized_model.bin", "model", model);

// Use the quantized model for inference
arma::mat testData; // Load your test data
arma::mat predictions;
model.Predict(testData, predictions);
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `Quantization` objects.
 * [`Apply()`](#applying-quantization): apply quantization to model or weights.
 * [Other functionality](#other-functionality) for customizing quantization parameters.
 * [Examples](#simple-examples) of simple usage and links to detailed example projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom behavior.
 * [Advanced template examples](#advanced-functionality-examples) of use with custom template parameters.

#### See also:

 * [`FFN`](ffn.md): feed-forward neural networks
 * [`CNN`](cnn.md): convolutional neural networks
 * [mlpack transformations](../../index.md#transformations)
 * [Quantization on Wikipedia](https://en.wikipedia.org/wiki/Quantization_(signal_processing))
 * [Quantization for Neural Networks](https://arxiv.org/abs/1712.05877) (comprehensive overview paper, pdf)

### Constructors

Construct a `Quantization` object using one of the constructors below. Defaults and types are detailed in the [Constructor Parameters](#constructor-parameters) section below.

#### Forms:

 * `q = Quantization()`
   - Initialize quantizer with default settings (8-bit linear quantization).

---

 * `q = Quantization(bits)`
   - Initialize quantizer with specified bit depth.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `bits` | `size_t` | Number of bits to use for quantization. | `8` |

### Applying Quantization

Once a `Quantization` object is created, the `Apply()` member function can be used to quantize models or weight matrices.

 * `q.Apply(model)`
   - Quantize the weights of the given `model` in-place.

---

 * `q.Apply(weights, quantizedWeights)`
   - Quantize the given `weights` matrix and store the result in `quantizedWeights`.

---

#### Quantization Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _model_ | `model` | `FFN<>` or `CNN<>` | Neural network model to be quantized. |
||||
| _weights_ | `weights` | [`arma::mat`](../matrices.md) | Weight matrix to be quantized. |
| _weights_ | `quantizedWeights` | [`arma::mat&`](../matrices.md) | Matrix to store quantized weights. |

### Other Functionality

 * A `Quantization` object can be serialized with [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `q.Bits()` will return a `size_t` indicating the number of bits used for quantization.

 * `q.SetBits(bits)` will set the number of bits used for quantization to `bits`.

For complete functionality, the [source code](/src/mlpack/methods/quantization/quantization.hpp) can be consulted. Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of `Quantization`.

---

Quantize a model to different bit depths and compare performance:

```c++
// Load a pre-trained model
mlpack::FFN<> model;
mlpack::data::Load("model.bin", "model", model);

// Load test data
arma::mat testData;
arma::mat testLabels;
mlpack::data::Load("test_data.csv", testData, true);
mlpack::data::Load("test_labels.csv", testLabels, true);

// Evaluate original model
double originalAccuracy = model.Evaluate(testData, testLabels);

// Quantize to 8-bit
mlpack::Quantization q8(8);
mlpack::FFN<> model8 = model;
q8.Apply(model8);
double accuracy8 = model8.Evaluate(testData, testLabels);

// Quantize to 4-bit
mlpack::Quantization q4(4);
mlpack::FFN<> model4 = model;
q4.Apply(model4);
double accuracy4 = model4.Evaluate(testData, testLabels);

std::cout << "Original accuracy: " << originalAccuracy << std::endl;
std::cout << "8-bit quantized accuracy: " << accuracy8 << std::endl;
std::cout << "4-bit quantized accuracy: " << accuracy4 << std::endl;
```

---

### Advanced Functionality: Template Parameters

The `Quantization` class supports template parameters for custom behavior. The full signature of the class is as follows:

```
Quantization<QuantizationStrategy = LinearQuantization,
             ModelType = FFN<>>
```

 * `QuantizationStrategy`: the strategy used for quantizing weights.
 * `ModelType`: the type of model to be quantized.

---

#### `QuantizationStrategy`

 * Specifies the algorithm used to quantize weights.
 * The `LinearQuantization` class is available and is the default.
 * A custom class must implement the following function:

```c++
class CustomQuantizationStrategy
{
 public:
  template<typename MatType>
  void QuantizeWeights(const MatType& weights, MatType& quantizedWeights)
  {
    // CUSTOM STRATEGY 
  }
};
```

### Advanced Functionality Examples

Use a custom quantization strategy:

```c++
```c++
class LogarithmicQuantization
{
 public:
  LogarithmicQuantization(size_t bits = 8) : bits(bits) {}

  template<typename MatType>
  void QuantizeWeights(const MatType& weights, MatType& quantizedWeights)
  {
    // Find the maximum absolute value in the weights
    double maxAbs = arma::max(arma::abs(weights));
    
    // Calculate the scaling factor
    double scale = std::pow(2, bits - 1) - 1;
    
    // Temporary matrix to store transformed values
    MatType logWeights = arma::sign(weights) % arma::log1p(arma::abs(weights) / maxAbs * std::exp(scale)) / scale;
    
    // Quantize to integers
    quantizedWeights = arma::round(logWeights * scale);
    
    // Clip values to ensure they're within the representable range
    quantizedWeights = arma::clamp(quantizedWeights, -scale, scale);
  }

  // Method to dequantize (for use during inference)
  template<typename MatType>
  void DequantizeWeights(const MatType& quantizedWeights, MatType& dequantizedWeights, double maxAbs)
  {
    double scale = std::pow(2, bits - 1) - 1;
    dequantizedWeights = maxAbs * (arma::exp(arma::abs(quantizedWeights) / scale) - 1) % arma::sign(quantizedWeights);
  }

 private:
  size_t bits;
};

// Create a quantizer with logarithmic quantization
mlpack::Quantization<LogarithmicQuantization> quantizer;

// Apply to a model
mlpack::FFN<> model;
mlpack::data::Load("model.bin", "model", model);
quantizer.Apply(model);
```