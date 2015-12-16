load data

%% logit

w = ((1:size(x,2))/5e4);
% w = ones(1,size(x,2));

sum(log(1 + exp ( - y .* (x * w'))))

g = full(x' * (-y ./ (1  + exp( y .* (x * w')))));
sum(g.*g)


%% fm

V_dim = 5;

w = (1:size(x,2))'/5e4;
V = w * (1:V_dim) / 10;

py = x * w + .5 * sum((x*V).^2 - (x.*x)*(V.*V), 2);

sum(log(1 + exp ( - y .* py)))

p = - y ./ (1 + exp (y .* py));

gw = x' * p;
gv = x' * bsxfun(@times, p, x*V) - bsxfun(@times, (x.*x)'*p, V);

g = [gw, gv];
sum(sum(g.*g))
