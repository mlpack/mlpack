% l2_fnorm() - perform an l2 norm in functional space
function norm = l2_fnorm(domain, f,g);

if size(f) == size(g')
  g = g';
end

norm = quad(@ppval, domain(1), domain(end), [], [], spline(domain, f .* g));
