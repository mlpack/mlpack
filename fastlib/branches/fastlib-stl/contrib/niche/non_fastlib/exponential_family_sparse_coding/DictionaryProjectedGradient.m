function D_opt = DictionaryProjectedGradient(D_0, S, T, alpha, beta)
%function D_opt = DictionaryProjectedGradient(D_0, S, T, alpha, beta)
%
% T are the sufficient statistics for the data, stored as
% num_features by num_points
%
% Nocedal and Wright suggested default alpha = 1e-4,
% but Boyd and Vandenberghe suggest using 0.01 to 0.3
%
% We use Armijo line search, with projection operator P, gradient g, and
% termination condition
%   f(P(x + t*g)) >= f(x) + alpha * g^T (P(x + t*g) - x)


if nargin == 3
  alpha = 1e-4;
  beta = 0.9;
elseif nargin == 4 
  beta = 0.9;
end

obj_tol = 1e-6;

d = size(D_0, 1);
n = size(S, 2);


for main_iteration = 1:1000

  fprintf('Main Iteration %d\n', main_iteration);

  sum_DS = exp(sum(D_0 * S));
  
  % compute gradient
  grad = zeros(size(D_0));
  for i = 1:n
    grad = grad - T(:,i) * S(:,i)';
  end
  
  for i = 1:n
%    grad = grad + repmat(S(:,i)', d, 1) * sum_DS(i);
    grad = grad + exp(D_0 * S(:,i)) * S(:,i)';
  end

  %grad = grad / n; % seems to be horrible
  grad = grad / sqrt(trace(grad' * grad)); % seems to work well
  
  % do line search along direction of negative gradient, using projected evaluation to find D_opt
  
  %start with step size t = 1, decreasing by beta until Armijo condition is satisfied
  %   f(P(x + t*g)) >= f(x) + alpha * g^T (P(x + t*g) - x)
  
  f_0 = ComputeDictionaryObjective(D_0, S, T);
  
  %fprintf('\t\t\t\ttrace(grad^T * grad) = %f\n', trace(grad' * grad));
  
  
  

  t = 1;
  iteration_num = 0;
  done = false;
  prev_best_f = f_0;
  while ~done
    iteration_num = iteration_num + 1;
    %fprintf('Iteration %d\n', iteration_num);

    D_t = D_0 - t * grad;
    norms = sqrt(sum(D_t .^ 2));
    % project if necessary
    for i = find((norms > 1))
      D_t(:,i) = D_t(:,i) / norms(i);
    end
    %disp(sum(D_t.^2));
    
    f_t = ComputeDictionaryObjective(D_t, S, T);
    %fprintf('f_0 = %f\tf_t = %f\n', f_0, f_t);
    %fprintf('\t\t\t\ttrace(grad^T * (D_t - D_0)) = %f\n', trace(grad' * (D_t - D_0))); 
    %fprintf('\t\t\t\tnorm(D_t - D_0) = %f\n', norm(D_t - D_0));

    if f_t <= f_0 + alpha * trace(grad' * (D_t - D_0))
      done = true;
      fprintf('\t\tObjective value: %f\n', f_t);
      if f_t > prev_best_f
	error('Objective increased! Aborting...');
	return;
      end
      if prev_best_f - f_t < obj_tol
	fprintf('Improvement to objective below tolerance. Finished.\n');
	D_opt = D_0;
	return;
      end
      prev_best_f = f_t;
      %fprintf('done!\n');
    end
    %pause;
    t = beta * t;
  end

  D_0 = D_t;
end

D_opt = D_0;