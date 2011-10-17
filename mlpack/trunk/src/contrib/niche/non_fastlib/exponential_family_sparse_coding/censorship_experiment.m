load ap_small;
tf_orig = tf;
load ap_small_dukakis_censored;
tf_cens = tf;

tf_orig = tf_orig(:,[1:952 954:end]);

tf_cens = tf_cens(:,[1:952 954:end]);

normalization_matrix = repmat(sqrt(sum(tf_orig.^2)), size(tf_orig, 1), 1);

tf_orig_normalized = tf_orig ./ normalization_matrix;
tf_cens_normalized = tf_cens ./ normalization_matrix;
clear normalization_matrix;




n_atoms = 10;

opts = statset('Display','iter', 'MaxIter', 3);
disp(opts);
[IDX C] = kmeans(tf_orig_normalized', n_atoms, 'Options', opts);
C_normalized = full(normcols(C'));

lambda = 0.2;
max_iter = 10;
[D S] = ExpFamSparseCoding('g', tf_orig_normalized, n_atoms, lambda, max_iter, ...
			   false,  C_normalized);

for j=1:n_atoms
  [y i] = sort(D(:,j));
  disp(vocab(i(end-9:end))');
end

max_iter_perturbed = 10;
lambda_Q = 0.3;

[Q1, S1] = ExpFamSparseCodingPerturbed('g', tf_cens_normalized, ...
				       n_atoms, lambda, lambda_Q, ...
				       max_iter_perturbed, false, D);
