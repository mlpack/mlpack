function f = ComputeDictionaryObjective(D, S, T)
%function f = ComputeDictionaryObjective(D, S, T)

n = size(S, 2);

f = 0;
f = f + trace(D' * T * S');
% easy to understand version of the above line
%for i = 1:n
%  f = f - S(:,i)' * D' * T(:,i);
%end

f = f + sum(sum(exp(D * S)));
% easy to understand version of the above line
%for i = 1:n
%  f = f + sum(exp(D * S(:,i)));
%end

f = f / n; % seems to work better with this normalization
