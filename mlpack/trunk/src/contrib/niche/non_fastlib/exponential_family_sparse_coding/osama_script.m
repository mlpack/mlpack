load /scratch/niche/text_data/json_osama_parser/tf_3docs.mat;
load /scratch/niche/text_data/json_osama_parser/vocab_3docs.mat;
tf_normalized = normcols(tf);

n_atoms = 20;

opts = statset('Display', 'iter');
[IDX C] = kmeans(tf_normalized', n_atoms, 'Options', opts);
C_normalized = full(normcols(C'));

lambda = 0.5;
max_iter = 100;
[D S] = ExpFamSparseCoding('g', tf_normalized, n_atoms, lambda, max_iter, ...
			   false,  C_normalized);

for j=1:10
  [y i] = sort(D(:,j));
  disp(vocab(i(end-9:end)));
end
