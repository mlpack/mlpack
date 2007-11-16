% find_opt_subrotation(X)
function [theta_star, rotator_star] = find_opt_subrotation(X);

thetas = linspace(-pi/4, pi/4, 200);

max_h_sum = -Inf;

for theta = thetas
  rotator =[cos(theta) -sin(theta); sin(theta) cos(theta)];
  rotated_X = rotator * X;
  
  h_sum = get_vasicek_entropy_estimate(rotated_X(1,:)) + ...
	  get_vasicek_entropy_estimate(rotated_X(2,:));
  
  if h_sum > max_h_sum
    max_h_sum = h_sum;
    theta_star = theta;
  end
end

rotator_star = ...
    [cos(theta_star) -sin(theta_star); ...
     sin(theta_star) cos(theta_star)];

theta_star
