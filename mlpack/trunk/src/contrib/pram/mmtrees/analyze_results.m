function analyze_results(filename)

results = load(filename);
[nPoints temp] = size(results);

error = 0; max_eps = -1.0; avg_eps = 0.0;
for i = 1:nPoints
  if results(i, 2) < results(i, 4)
    error = error + 1;
    epsilon = (results(i,4) / results(i,2)) - 1;
    if epsilon > max_eps
      max_eps = epsilon;
    end
    avg_eps = avg_eps + epsilon;
  end
end
avg_eps = avg_eps / nPoints;
display(sprintf('Error:%d/%d, Avg. Eps:%f, Max Eps:%f',...
		error, nPoints, avg_eps, max_eps));

clear all;
