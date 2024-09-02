# Post-Training Quantization

The quantization is used to reduce the  the precision of the weights and activations to other (lower) bit-widths. This is usually useful when working on limited hardware, where a reduced model size is required.

You can choose your conversion factor with the `LinearQuantization` and `ScaleQuantization `functions. They both map the weights from floating-point weights to the int8 range, but they use different ways to calculate and apply the conversion.

**Linear Quantization:**
The scale calculation goes as follows:
$$
\text{scale} = \frac{(2^{\text{numBits} - 1} - 1)}{\text{maxAbs}}
$$

It is scaling the weights so that the maximum aboslute value fits exactly to the maximum int8 value (`127` for 8 bits).
It multiplies the weights by the scale and clamps them to fit into the in8 range. So the scaling occurs before clamping, so that squeezing the weights into the range by scaling them up. 
When using this be aware that it maximizes the use of the int8 range, so it is a bit more "aggressive".

**Scale Quantization:**
The scale calculation goes as follows:
$$
\text{scale} = \frac{\text{maxAbs}}{(2^{\text{numBits} - 1} - 1)}
$$

The weights are divided by this scale factor and then rounded to fit within the int8 range. By using this method, the scale adjusts so  that when the floating-point weights are divided by it, the values fall directly within the int8 range.
The weights are scaled down before rounding, so it will lead to a different distribution of quantized values compared to multiplication. Since it divides the weights to fit within the range it is a more "conservative" approach because it has less precision loss due to the rounding.


### Usage Example

```cpp
#include <iostream>
#include <mlpack.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

int main() {

    // Create a simple Feedforward Neural Network (FFN) model.
    FFN<NegativeLogLikelihood, GlorotInitialization, arma::mat> model;
    
    // Add a Linear layer with 5 input units and 3 output units.
    model.Add<Linear>(5, 3); 

    // Initialize model parameters with random values.
    model.Parameters().randu();

    std::cout << "Initial model parameters:\n" << model.Parameters() << std::endl;

    // Quantize the entire network to use integer parameters (default Linear Quantization).
    FFN<NegativeLogLikelihood, GlorotInitialization, arma::imat> quantizedModel = model.Quantize<arma::imat>();

    std::cout << "Quantized model parameters:\n" << quantizedModel.Parameters() << std::endl;
    std::cout << "Minimum value in quantized parameters: " << quantizedModel.Parameters().min() << std::endl;
    std::cout << "Maximum value in quantized parameters: " << quantizedModel.Parameters().max() << std::endl;

    return 0;
}
```

