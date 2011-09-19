function f = ComputeDictionaryObjective(D, S, T)
%function f = ComputeDictionaryObjective(D, T)

f = 0;
for i = 1:m
  f = f + S(:,i)' * D' * T(:,i);
end

for i = 1:m
  f = f + sum(exp(B * S(:,i)));
end
