% h_kde() - produce an entropy estimate using the entropy of a
% kernel density estimate of the true distribution
function H = h_kde(N,h);

x = normrnd(zeros(N,1), 1);



sum_logs = 0;
for i=1:N
  sum = 0;
  for j = 1:N
    sum = sum + exp(-((x(i) - x(j))^2) / (2*(h^2)));
  end
  sum_logs = sum_logs + log(sum);
end

H = log(N*h*sqrt(2*pi)) - (sum_logs / N);
