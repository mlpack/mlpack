X = load('swiss1500_csdp.sol_X');
X = X(:,3:end);
sparse_K = spconvert(X);
K = full(sparse_K);
K = K + K' - diag(diag(K));
[V,D] = eig(K);
plot(diag(D));
scatter(V(:,end-1), V(:,end));
%scatter3(V(:,end-2), V(:,end-1), V(:,end));
