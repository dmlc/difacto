load data

%% logit

w = ((1:size(x,2))/5e4);
% w = ones(1,size(x,2));

sum(log(1 + exp ( - y .* (x * w'))))

g = full(x' * (-y ./ (1  + exp( y .* (x * w')))));
sum(g.*g)

tau = 1 ./ (1  + exp( y .* (x * w')));
h = full((x.*x)' * (tau .* (1-tau)));
sum(h.*h)


%% fm

V_dim = 5;

w = (1:size(x,2))'/5e4;
V = w * (1:V_dim) / 10;

[objv, gw, gV] = fm_loss(y, x, w, x, V);

objv
sum(gw.^2) + sum(gV(:).^2)
