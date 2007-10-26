% kde_unbiased() - produces evaluations of the kde of the pdf of the data
function [f_hat_X] = kde_unbiased(X, h);

N = length(X);



f_hat_X = zeros(size(X));

for i = 1:N
  x_i = X(i);
  f_hat_x_i = 0;
  for j = 1:N
    f_hat_x_i = f_hat_x_i + exp(-((x_i - X(j))^2) / ( 2 * (h^2)) );
  end
  f_hat_X(i) = f_hat_x_i;
end

f_hat_X = f_hat_X - 1;

f_hat_X = f_hat_X / ( (N - 1) * h * sqrt(2 * pi) );
