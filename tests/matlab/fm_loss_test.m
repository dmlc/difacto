load data

%% logit

w = ((1:size(x,2))/5e4);
% w = ones(1,size(x,2));

sum(log(1 + exp ( - y .* (x * w'))))

g = full(x' * (-y ./ (1  + exp( y .* (x * w')))));
sum(g.*g)


%% fm
