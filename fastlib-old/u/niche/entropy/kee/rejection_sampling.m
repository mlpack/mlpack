S = 22 * rand(10000,1) - 10;

N = length(S);

p_S = normpdf(S, 5, 1);

collect = 0;
count = 0;

for i = 1:N
  if rand < p_S(i)
    collect = collect + S(i);
    count = count + 1;
  end  
end

collect / count
