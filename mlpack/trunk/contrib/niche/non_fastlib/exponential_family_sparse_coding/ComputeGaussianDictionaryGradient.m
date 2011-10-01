function grad = ComputeGaussianDictionaryGradient(D, S, T)
%function grad = ComputeGaussianDictionaryGradient(D, S, T)

grad = -T * S' + D * S * S';
