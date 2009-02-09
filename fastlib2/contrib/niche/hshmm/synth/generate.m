n = 2;
d = 1;

mu_0 = rand(n,1);
Sigma_0 = random_cov(n);

A = rand(n);
A = A / max(eig(A));

C = rand(d, n);

mu_w = zeros(1, n);
Q = random_cov(n);

mu_v = zeros(1, d);
R = random_cov(d);


T = 10000;

x = zeros(n, T);
y = zeros(d, T);

x(:, 1) = mvnrnd(mu_0, Sigma_0);
y(:, 1) = C * x(:, 1) + mvnrnd(mu_v, R);
for t = 2:T
  x(:, t) = A * x(:, t-1) + mvnrnd(mu_w, Q)';
  y(:, t) = C * x(:, t) + mvnrnd(mu_v, R)';
end


mu_0 = mu_0';
save mu_0.dat mu_0 -ascii;

Sigma_0 = Sigma_0';
save Sigma_0.dat Sigma_0 -ascii;

A = A';
save A.dat A -ascii;

C = C';
save C.dat C -ascii;

Q = Q';
save Q.dat Q -ascii;

R = R';
save R.dat R -ascii;

x = x';
save X.dat x -ascii;

y = y';
save Y.dat y -ascii;
