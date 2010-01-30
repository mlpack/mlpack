X = load('distance_constraints.csv', 'r');

X(:,3) = X(:,3) / max(X(:,3));

% The first row of X should contain:
%  (number of points) (number of constraints) -1

n_points = X(1, 1);
n_constraints = X(1, 2);

X = X(2:end,:);

for k = 1:size(X, 1)
  if X(k,1) > X(k,2)
    temp = X(k,1);
    X(k,1) = X(k,2);
    X(k,2) = temp;
  end
end


X = sortrows(X, [1 2]);


n_constraints = 0;

last_i = -1;
last_j = -1;
for k = 1:size(X, 1)
  i = X(k, 1);
  j = X(k, 2);
  
  if (i ~= last_i) || (j ~= last_j)
    n_constraints = n_constraints + 1;
  end
  last_i = i;
  last_j = j;
end



fid = fopen('upper_triangularized_distance_constraints.csv', 'w');

fprintf(fid, '%d %d\n', n_points, n_constraints);

last_i = -1;
last_j = -1;
for k = 1:size(X, 1)
  i = X(k, 1);
  j = X(k, 2);
  
  if (i ~= last_i) || (j ~= last_j)
    fprintf(fid, '%d %d %.18e\n', i, j, X(k, 3));
  end
  last_i = i;
  last_j = j;
end

fclose(fid);

  
  
   
