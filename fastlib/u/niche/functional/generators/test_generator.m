function [x,y] = test_generator();

N = 1e6;

x = zeros(N,1);
y = zeros(N,1);

for i=1:N
  x(i) = laplacinv(rand, .5, .5);
  y(i) = laplacinv(rand, .5, .5);
end
