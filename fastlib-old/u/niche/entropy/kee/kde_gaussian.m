% kde() - produces evaluations of the kde of the pdf of a normal
% random variable
function [p_y, mu, sigma] = kde_gaussian(Y, N, h);

x = normrnd(zeros(N,1), 1);

mu = mean(x);
sigma = std(x);

%disp('done generating data');

p_y = zeros(size(Y));

i = 1;
for y=Y
  sum = 0;
  for j = 1:N
    sum = sum + exp(-((y - x(j))^2) / (2*(h^2)));
  end
  p_y(i) = sum;
  i = i + 1;
end

p_y = p_y / (N*h*sqrt(2*pi));
