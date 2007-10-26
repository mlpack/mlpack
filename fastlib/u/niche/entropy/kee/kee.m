% kee() - kernel entropy estimation
function [second_term] = kee();
% select bandwidth h_E that minimizes LOO cross-entropy (expectation of
% log of kernel density estimate over the empirical distribution
% function)

% search for bandwidth that minimizes:
%  \int{(f_hat(x) * log(f_hat(x)))^2 dx}
%  - 2 E[f_hat(x) * log(f_hat(x))]
% evaluate \int{(f_hat(x) * log(f_hat(x)))^2 dx}

% calculate the LOOCV kernel bandwidth
% over log

% first calculate kernel sum

N = 1000;

X = normrnd(zeros(N,1),1);

h_E = find_h_E(X);
f_hat_h_E_X = kde_unbiased(X, h_E);

h_set = linspace(.02,1,50);


second_term = zeros(size(h_set));

for i = 1:length(h_set)
  h = h_set(i);
  h
  f_hat_h_X = kde_unbiased(X, h);
  
  second_term(i) = sum(logp(f_hat_h_E_X) .* f_hat_h_X .* logp(f_hat_h_X)) / N;  
end  
  

% test each bandwidth
% for each data point x_i
%    calculate:
%        (1) log f_hat_h_E(x_i)
%        (2) f_hat_h(x_i)
%        (3) log f_hat_h(x_i)
%    take the product of (1),(2),(3)
% end
