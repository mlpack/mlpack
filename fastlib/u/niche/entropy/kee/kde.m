% kde() - produces evaluations of the kde of the pdf of the data
function [p_X] = kde(X, h);

N = length(X);



p_X = zeros(size(X));

i = 1;
for x=X'
  sum = 0;
  for j = 1:N
    sum = sum + exp(-((x - X(j))^2) / (2*(h^2)));
  end
  p_X(i) = sum;
  i = i + 1;
end

p_X = p_X / (N*h*sqrt(2*pi));

