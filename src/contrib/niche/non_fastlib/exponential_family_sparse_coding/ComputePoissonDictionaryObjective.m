function f = ComputePoissonDictionaryObjective(D, S, T)
%function f = ComputePoissonDictionaryObjective(D, S, T)

n = size(S, 2);

f = -trace(D' * T * S') + sum(sum(exp(D * S))); % possibility of numerical overflow

f = f / n; % seems to work better with this normalization


% Note that a = -trace(D' * T * S') is equivalent to:
%   a = 0;
%   for i = 1:n
%     a = a - S(:,i)' * D' * T(:,i);
%   end
%
% Also note that a = sum(sum(exp(D * S))) is equivalent to:
%   a = 0;
%   for i = 1:n
%     a = a + sum(exp(D * S(:,i)));
%   end


%f = 0;
%f = f - trace(D' * T * S');
%f = f + sum(sum(exp(D * S)));
