%{
assumptions:
  f(x) ~ N(mu, sigma)
%}


Y = linspace(-10,10,1e4);

% sample at points f(x) * log f(x)
f_Y = normpdf(Y);


% use samples to approximate f(x) log(f(x))
H_Y = sum(f_Y .* logp(f_Y)) / sum(f_Y)

% weight the product of f^ log f^ by our approximation to f log f


% set kernel parameter(s)
%h = .5;

% simulate some data
%X = normrnd(zeros(1,1000), 1);
N = length(X);

% get f^(y)
f_hat_Y = zeros(size(Y));

i = 1;
for y = Y
  collect = 0;
  for j = 1:N
    collect = collect + exp(-((y - X(j))^2) / (2*(h^2)));
  end
  f_hat_Y(i) = collect;
  i = i + 1;
end

f_hat_Y = f_hat_Y / (N*h*sqrt(2*pi));

terms = zeros(1,3);

terms(1) = sum((f_Y .* logp(f_Y)) .^ 2) / sum(f_Y);

terms(2) = -2 * sum(f_Y .* logp(f_Y) .* f_hat_Y .* logp(f_hat_Y)) / sum(f_Y);

terms(3) = sum((f_hat_Y .* logp(f_hat_Y)) .^ 2) / sum(f_hat_Y);

sum(terms)


