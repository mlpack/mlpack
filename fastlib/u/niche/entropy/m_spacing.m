% m_spacing() - estimate entropy using m-spacing
function H = m_spacing(X, m);

X = X / std(X);

N = length(X);

Z = sort(X, 'ascend');

sum = 0;
for i = 1 : (N - m)
  sum = sum + log2( (N / m) * (Z(i + m) - Z(i)) );
end

H = sum / N;