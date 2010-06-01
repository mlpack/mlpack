function sigma = random_cov(d);

rand_vecs = normrnd(0, 1, d, d);
for i = 1:(d-1)
  for j = (i+1):d
    rand_vecs(i,j) = rand_vecs(j,i);
  end
end

[V, D] = eig(rand_vecs);

for i = 1:d
  V(:,i) = V(:,i) / norm(V(:,i));
end

for i = 1:d
  D(i,i) = rand;
end

sigma = V * D * V';
