load ap_small;
tf_orig = tf;
load ap_small_dukakis_censored;
%load ap_small_bush_half_censored;
tf_cens = tf;

tf_orig = tf_orig(:,[1:952 954:end]);

tf_cens = tf_cens(:,[1:952 954:end]);

normalization_matrix = repmat(sqrt(sum(tf_orig.^2)), size(tf_orig, 1), 1);

tf_orig_normalized = tf_orig ./ normalization_matrix;
tf_cens_normalized = tf_cens ./ normalization_matrix;
clear normalization_matrix;



n_atoms = 10;

opts = statset('Display','iter', 'MaxIter', 10);
disp(opts);
[IDX C] = kmeans(tf_orig_normalized', n_atoms, 'Options', opts);
C_normalized = full(normcols(C'));

lambda = 0.2;
max_iter = 100;

return;

[D S] = ExpFamSparseCoding('g', tf_orig_normalized, n_atoms, lambda, max_iter, ...
			   false,  C_normalized);

for j=1:n_atoms
  [y i] = sort(D(:,j));
  disp(vocab(i(end-9:end))');
end

%max_iter_perturbed = 100;
%lambda_Q = 0.3;

% settings which show recognition of dukakis censoring
max_iter_perturbed = 20;
lambda_Q = 0.2;

[Q_orig, S_orig] = ExpFamSparseCodingPerturbed('g', tf_orig_normalized, n_atoms, lambda, lambda_Q, max_iter_perturbed, false, D);
[Q_cens, S_cens] = ExpFamSparseCodingPerturbed('g', tf_cens_normalized, n_atoms, lambda, lambda_Q, max_iter_perturbed, false, D);
