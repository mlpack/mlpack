load ../sparse_censorship/take_two_laplace_regularizer/tf_small.mat;
tf = tf(1:1000,:);
tf = tf(:,find(sum(tf) > 1))';


