% objective_function() - an objective function
function C = objective_function(x, a);

% C = x.^4 - 100 * x.^3 - 1000 * x.^2 - x + 1;

C = (a - x)^2;
