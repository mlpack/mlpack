% logp() - log for probabilities - if x = 0 -> 0, else -> log(x)
function [l] = logp(x);

N = length(x);
l = zeros(size(x));

for i=1:N
  if x(i) == 0
    l(i) = 0;
  else
    l(i) = log(x(i));
  end
end


% how bad of an approximation is int(x f(x)) = (1/n)
% \sum_{i=1}^nx_i

% assume random draws using some pdf - what is the rate of
% convergence of the error in expectated value using the empirical
% distribution function
