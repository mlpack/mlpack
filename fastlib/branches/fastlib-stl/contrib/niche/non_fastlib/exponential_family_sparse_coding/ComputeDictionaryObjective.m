function f = ComputeDictionaryObjective(D, S, T)
%function f = ComputeDictionaryObjective(D, S, T)

n = size(S, 2);

f = 0;
for i = 1:n
  f = f - S(:,i)' * D' * T(:,i);
end

for i = 1:n
%  f = f + exp(sum(D * S(:,i)));
  f = f + sum(exp(D * S(:,i)));
end

f = f / n; % seems to work better with this normalization
