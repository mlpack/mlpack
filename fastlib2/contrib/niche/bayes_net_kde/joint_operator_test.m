function [hs_norm] = joint_operator_test(A, B)

m = length(A);

h = 1;

I = eye(m);

lambda = 1e-6;%1e-12;

H = I - (1/m * ones(m, 1) * ones(1, m));

KA = zeros(m);
KB = zeros(m);
for i = 1:m
  for j = 1:m
    KA(j, i) = exp(-((A(i) - A(j))^2) / h);
    KB(j, i) = exp(-((B(i) - B(j))^2) / h);
  end
end

JA = inv(H * KA + lambda * m * I) * H;

JAji = JA * KA;



% compute inner product of DAG embedding with DAG embedding

message_ip_j = zeros(m, m);
for ip = 1:m
  for j = 1:m
    jp_sum = 0;
    for jp = 1:m
      jp_sum = jp_sum + JAji(jp, ip) * KB(jp, j);
    end
    message_j_ip(j, ip) = jp_sum;
  end
end

sum = 0;
for i = 1:m
  for ip = 1:m
    if ip == i
      continue;
    end
    j_sum = 0;
    for j = 1:m
      j_sum = j_sum + JAji(j, i) * message_j_ip(j, ip);
    end
    sum = sum + KA(ip, i) * j_sum;
  end
end
dag_dot_dag = sum / (m * (m-1))



% compute inner product of dag embedding with full embedding

sum = 0;
for l = 1:m
  for i = 1:m
    if i == l
      continue;
    end
    j_sum = 0;
    for j = 1:m
      j_sum = j_sum + JAji(j, i) * KB(j, l);
    end
    sum = sum + KA(i, l) * j_sum;
  end
end
dag_dot_full = sum / (m * (m-1))



%compute inner product of full embedding with full embedding

sum = 0;
for i = 1:m
  for ip = 1:m
    if ip == i
      continue;
    end
    sum = sum + KA(ip, i) * KB(ip, i);
  end
end
full_dot_full = sum / (m * (m-1))


hs_norm = dag_dot_dag - 2 * dag_dot_full + full_dot_full



%{
sum = 0;
for i = 1:m
  for ip = 1:m
    j_sum = 0;
    for j = 1:m
      for jp = 1:m
	j_sum = j_sum + JAji(j,i) * JAji(jp,ip) * KB(j, jp);
      end
    end
    sum = sum + KA(i, ip) * j_sum;
  end
end

dag_dot_1 = sum / m^2


sum = 0;
for i = 1:m
  sum = sum + KA(i, 1) * KB (i, 1);
end
full_dot_1 = sum / m
%}
