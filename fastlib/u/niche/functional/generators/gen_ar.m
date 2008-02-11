function [x] = gen_ar(p, T);

rand('state', sum(100*clock));

% time kernel
%a = rand(1,p); % = [a_p a_{p-1} ... a_1]
for i = 1:p
  a(i) = .9^(p - i);
end
a = a * 1 / sum(a);

%a = sort(a);

% data
x = zeros(T,1); % = [x(1) x(2) ... x(T)]'

x(1:p) = rand;
for t = (p+1):T
  e = normrnd(0,.01);
  x(t) = a * x((t - p):(t - 1)) + e;
end
