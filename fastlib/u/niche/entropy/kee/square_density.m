X = normrnd(zeros(10000,1),1);

h = .2;


f_hat_h_X = kde_unbiased(X,h);

mean(f_hat_h_X)
  
