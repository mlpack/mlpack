
% initialize random number generator
rand('state', sum(100*clock));


% sample from laplacian

D = 2;
N = 10000;

mu = 3;
sigma = 1;
b = sqrt(sigma/2);

clear S;

for i = 1:D
  S(i,:) = laplacinv(rand(N, 1), mu, b);
end

length(find(S < 0))
while 1
  neg_indices = find(S < 0);

  len_neg_indices = length(neg_indices);

  if len_neg_indices > 0
    S(neg_indices) = laplacinv(rand(len_neg_indices, 1), mu, b);
  else
    break;
  end
end

% impose unit variance on each row of S

for i = 1:D
  S(i,:) = S(i,:) / std(S(i,:));
end


% X = A S
% set mixing matrix A

A = rand(D);


X = A * S;


A_true = A;
S_true = S;

% factorize X into 2 matrices A and S
% initialize A to the identity matrix and S to X

A = eye(D);
S = X;


% perturb S such that S is still positive, and find a new matrix A
% what is the class of perturbations that are admissible by S?
% given a perturbation of matrix S, how can we go about finding A

% Algorithm 1
% 1 Initialize A to I and S to X
% 2 Perturb S to increase that statistical independence of its components
% 3 Find A that minimizes the Frobenius norm of X - A S


% Algorithm 2
% 1 Initialize A to I and S to X
% 2 Find A that maximizes entropy of inv(A) * X and minimizes
% negativity of inv(A) * X


h = get_vasicek_entropy_estimate(S);


% first, try unconstrained nonlinear optimization

bestW = fminunc(@(W_param) obj(W_param, X), eye(D));








