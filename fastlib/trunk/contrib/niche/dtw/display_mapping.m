function display_mapping(x1, x2, map)

t1 = 1:length(x1);
t2 = 1:length(x2);

vshift_x2 = x2 - (max(x2) - min(x1));

max_hshift = max(abs(diff(map, [], 2)));

figure(1);
clf;
hold on;
for i = 1:size(map, 1)
  rel_hshift = abs(t1(map(i,1)) - t2(map(i,2))) / max_hshift;
  
  plot([t1(map(i,1)) t2(map(i,2))], ...
       [x1(map(i,1)) vshift_x2(map(i, 2))], ...
       'Color', 1 - (rel_hshift * [0 1 1]));
  
end
plot(t1, x1, 'b');
plot(t2, vshift_x2, 'Color', [0 0.7 0]);

