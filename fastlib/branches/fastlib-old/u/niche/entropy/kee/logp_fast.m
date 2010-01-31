% logp_fast() - log for probabilities - if x = 0 -> 0, else -> log(x)
function [l] = logp_fast(x);

l = 0;

if x == 0
  l = 0;
else
  l = log(x);
end
