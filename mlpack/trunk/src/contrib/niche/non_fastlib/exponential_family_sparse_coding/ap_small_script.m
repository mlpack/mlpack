load ap_small;

%document 953 is all zeros, so remove it
tf = tf(:,[1:952 954:end]);

tf_normalized = normcols(tf);

n_atoms = 10;

opts = statset('Display','iter', 'MaxIter', 30);
disp(opts);
disp('hi');
%opts = statset('Display', 'iter');
[IDX C] = kmeans(tf_normalized', n_atoms, 'Options', opts);
C_normalized = full(normcols(C'));

lambda = 0.2;
max_iter = 100;
[D S] = ExpFamSparseCoding('g', tf_normalized, n_atoms, lambda, max_iter, ...
			   false,  C_normalized);

for j=1:n_atoms
  [y i] = sort(D(:,j));
  disp(vocab(i(end-9:end)));
end
