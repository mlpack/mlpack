load_tiny_rcv1;
tf_binary = double(tf > 0);

k = 10;
lambda = 1;

n_iterations = 10;

[D, S] = ExpFamSparseCoding('b', tf_binary, k, lambda, n_iterations, ...
			    false);
save Bernoulli_k10_lambda10_tracenormalization_c100 D S n_iterations lambda k;
