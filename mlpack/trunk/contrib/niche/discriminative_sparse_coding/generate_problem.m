d = 200;
k = 100;
n = 10000;

D = normrnd(0,1,d,k);
D = D ./ repmat(sqrt(sum(D.^2)), d, 1); % normalize each column

S = normrnd(0,1,k,n);
for i = 1:n
  permuted_inds = randperm(100);
  zero_inds = permuted_inds(1:90);
  S(zero_inds,i) = 0;
end

w = normrnd(0,1,k,1);
w = w / norm(w);

y = S' * w;

a = sort(abs(y), 'descend');
thousand_mark = a(1000);

good_inds = find(abs(y) >= thousand_mark);

S = S(:, good_inds);
y = sign(y(good_inds));
n = 1000;

X = D * S;
X = X + normrnd(0,1, size(X));

save X_hard.dat X -ascii;
save y_hard.dat y -ascii;
save w_hard.dat w -ascii;
save D_hard.dat D -ascii;
save S_hard.dat S -ascii;
save hard_example X y w D S;
