function D_opt = DictionaryProjectedGradient(D_0, S, T, alpha, beta)
%function D_opt = DictionaryProjectedGradient(D_0, S, T, alpha, beta)
%
% T are the sufficient statistics for the data, stored as
% num_features by num_points
%
% Nocedal and Wright suggested default alpha = 1e-4
%
% We use Armijo line search, with projection operator P, gradient g, and
% termination condition
%   f(P(x + t*g)) >= f(x) + alpha * g^T (P(x + t*g) - x)


if nargin == 3
  alpha = 1e-4;
  beta = 0.99;
elseif nargin == 4 
  beta = 0.99;
end

d = size(D_0, 1);

sum_DS = exp(sum(D_0 * S));

% compute gradient
grad = zeros(D_0);
for i = 1:m
  grad = grad + T(:,i) * S(:,i)';
end

for i = 1:m
  grad = grad + repmat(S(:,i)', d, 1) * sum_DS(i);
end

% do line search along direction of negative gradient, using projected evaluation to find D_opt

%start with step size t = 1, and slowly decrease it until Armijo
%condition is satisfied
%   f(P(x + t*g)) >= f(x) + alpha * g^T (P(x + t*g) - x)

f_0 = ComputeDictionaryObjective(D_0, S, T);

t = 1;
D_t = D_0 - t * grad;
norms = sum(D_t .^ 2);
for i = find((norms > 1))
  D_t(:,i) = D_t(:,i) / norms(i);
end

f_t = ComputeDictionaryObjective(D_t, S, T);


while f_t < f_0 + alpha * trace(grad' * (D_t - D_0))
  t = beta * t;

  D_t = D_0 - t * grad;
  norms = sum(D_t .^ 2);
  % project if necessary
  for i = find((norms > 1))
    D_t(:,i) = D_t(:,i) / norms(i);
  end
  
  f_t = ComputeDictionaryObjective(D_t, S, T);
end

D_opt = D_t;
