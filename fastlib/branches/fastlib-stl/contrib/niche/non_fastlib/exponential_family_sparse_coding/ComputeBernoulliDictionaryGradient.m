function grad = ComputeBernoulliDictionaryGradient(D, S, T)
%function grad = ComputeBernoulliDictionaryGradient(D, S, T)

grad = -T * S' + (1 ./ 1 + exp(-D * S)) * S'
