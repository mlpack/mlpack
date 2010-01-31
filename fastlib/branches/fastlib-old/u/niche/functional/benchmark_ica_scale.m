

for i = 1:12
  YoptNormed(i,:) = Yopt(i,:) / norm(Wopt(i,:));
end

for i = 1:12
  WoptNormed(i,:) = Wopt(i,:) / norm(Wopt(i,:));
end

YoptNormed_check = WoptNormed * score(:,1:p_small)';

WoptNormed

ic_components = WoptNormed * coeff(:,1:p_small)';
t = linspace(0, 1, T);
plot(t, -ic_components(6,:))
