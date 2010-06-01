function [hs_norm] = dag_accuracy(A, B, C)

m = length(A);

h = 1;

I = eye(m);

lambda = 1e1;

H = I - (1/m * ones(m, 1) * ones(1, m));

KA = zeros(m);
KB = zeros(m);
KC = zeros(m);
for i = 1:m
  for j = 1:m
    KA(j, i) = exp(-h * (A(i) - A(j))^2);
    KB(j, i) = exp(-h * (B(i) - B(j))^2);
    KC(j, i) = exp(-h * (C(i) - C(j))^2);
  end
end

JA = inv(H * KA + lambda * I);
JB = inv(H * KB + lambda * I);
JC = inv(H * KC + lambda * I);


JAji = JA * KA;
JBkj = JB * KB;

message_jp_k = zeros(m, m);
for jp = 1:m
  for k = 1:m
    kp_sum = 0;
    for kp = 1:m
      kp_sum = kp_sum + JBkj(kp,jp) * KC(k,kp);
    end
    message_jp_k(jp, k) = kp_sum;
  end
end

message_j_jp = zeros(m, m);
for j = 1:m
  for jp = 1:m
    k_sum = 0;
    for k = 1:m
      k_sum = k_sum + JBkj(k,j) * message_jp_k(jp, k);
    end
    message_j_jp(j, jp) = k_sum;
  end
end

message_ip_j = zeros(m, m);
for ip = 1:m
  for j = 1:m
    jp_sum = 0;
    for jp = 1:m
      kp_sum = kp_sum + JAji(jp, ip) * KB(j,jp) * message_j_jp(j, jp);
    end
    message_ip_j(ip, j) = kp_sum;
  end
end

dag_dot_dag = 0;
for i = 1:m
  for ip = 1:m
    j_sum = 0;
    for j = 1:m
      j_sum = JAji(j, i) * message_ip_j(ip, j);
    end
    dag_dot_dag = dag_dot_dag + KA(i, ip) * j_sum;
  end
end
dag_dot_dag = dag_dot_dag / m^6;      





for l = 1:m
  for j = 1:m
    k_sum = 0;
    for k = 1:m
      k_sum = k_sum + JBkj(k, j) * KC(k, l);
    end
    message_lj(l, j) = k_sum;
  end
end

full_dot_dag = 0;
for l = 1:m
  for i = 1:m
    j_sum = 0;
    for j = 1:m
      j_sum = j_sum + JAji(j, i) * KB(j, l) * message_lj(l, j);
    end
    full_dot_dag = full_dot_dag + KA(i, l) * j_sum;
  end
end
full_dot_dag = full_dot_dag / m^4;


full_dot_full = 0;
for i = 1:m
  for ip = 1:m
    full_dot_full = ...
	full_dot_full + KA(i, ip) * KB(i, ip) * KC(i, ip);
  end
end
full_dot_full = full_dot_full / m^2;

full_dot_full
full_dot_dag
dag_dot_dag

hs_norm = full_dot_full - 2 * full_dot_dag + dag_dot_dag;