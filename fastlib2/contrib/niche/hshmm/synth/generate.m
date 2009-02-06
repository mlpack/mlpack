n = 2;
d = 1;

A = rand(d);
A = A / max(eig(A));

C = rand(d, n);

mu_w = zeros(1, n);
Q = random_cov(n);

mu_v = zeros(1, d);
R = random_cov(d);


T = 10000;

x = zeros(n, T);
y = zeros(d, T);

x(:, 1) = mvnrnd(mu_w, Q);
y(:, 1) = C * x(:, 1) + mvnrnd(mu_v, R);
for t = 2:T
  x(:, t) = A * x(:, t-1) + mvnrnd(mu_w, Q)';
  y(:, t) = C * x(:, t) + mvnrnd(mu_v, R)';
end
