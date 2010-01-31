% kde() - evaluate the density of the kde induced by data X and points Y
function [p_Y] = eval_kde(X, Y, h);

N = length(X);

M = length(Y);



p_Y = zeros(size(Y));

for i = 1:M
  p_Y_i = 0;
  for j = 1:N
    p_Y_i = p_Y_i + exp(-((Y(i) - X(j))^2) / (2*(h^2)));
  end
  p_Y(i) = p_Y_i;
end

p_Y = p_Y / (N * h * sqrt(2 * pi));
