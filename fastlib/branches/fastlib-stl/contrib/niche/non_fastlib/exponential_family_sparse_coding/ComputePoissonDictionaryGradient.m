function grad = ComputePoissonDictionaryGradient(D, S, T)
%function grad = ComputePoissonDictionaryGradient(D, S, T)

grad = -T * S' + exp(D * S) * S'; % possibility of numerical overflow

% Note that A = -T * S' is equivalent to:
%   A = zeros(size(D));
%   for i = 1:n
%     A = A - T(:,i) * S(:,i)';
%   end
%
% Also note that A = exp(D * S) * S' is equivalent to
%   A = zeros(size(D));
%   for i = 1:n
%     A = A + exp(D_0 * S(:,i)) * S(:,i)';
%   end
