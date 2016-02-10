% load rcv1

%%
load data
[i,j,k]=find(x);
z = full(sum(sparse(i,j,ones(size(i))))>0);
X = x(:,z);
Y = y;

%%

grad = @(y, X, w) full(X' * (-y ./ (1  + exp(y .* (X * w)))));
objv = @(y, X, w) sum(log(1 + exp ( - y .* (X * w))));

max_m = 5;
c1 = 1e-4;
c2 = .9;
rho = .5;

[n, p] = size(X);
w = zeros(p,1);
s = [];
y = [];
g = grad(Y, X, w);

for k = 1 : 20
% two loop
  m = size(y, 2);
  p = - g;
  alpha = zeros(m,1);
  for i = m : -1 : 1
    alpha(i) = (s(:,i)' * p ) / (s(:,i)' * y(:,i) + 1e-10);
    p = p - alpha(i) * y(:,i);
  end
  if m > 0
    p = (s(:,m)'*y(:,m)) / (y(:,m)'*y(:,m) + 1e-10) * p;
  end
  for i = 1 : m
    beta = (y(:,i)'*p) / (s(:,i)'*y(:,i));
    p = p + (alpha(i) - beta) * s(:,i);
  end
  p = min(max(p, -5), 5);

% back tracking
  o = objv(Y, X, w);
  alpha = 1;
  gp = g'*p;
  fprintf('epoch %d, objv %f, gp %f\n', k, o, gp);
  for j = 1 : 10
    new_o = objv(Y, X, w+alpha*p);
    new_gp = grad(Y, X, w+alpha*p)' * p;
    fprintf('alpha %f, new_objv %f, new_gp %f\n', alpha, new_o, new_gp);
    if (new_o <= o + c1 * alpha * gp) && (new_gp >= c2 * gp)
      break;
    end
    alpha = alpha * rho;
  end

  if m == max_m
    s = s(:,2:m);
    y = y(:,2:m);
  end
  w = w + alpha * p;
  old_g = g;
  g = grad(Y, X, w);
  s = [s, alpha*p];
  y = [y, g - old_g];
end
