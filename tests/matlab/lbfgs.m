%%
load data
[i,j,k]=find(x);
cnt = full(sum(sparse(i,j,ones(size(i)))));
X = x(:,cnt>0);
Y = y;
X2 = x(:,cnt>0);

%%
V_dim = 5;
if V_dim == 0
  X2 = [];
end

V = repmat(((1:V_dim) - V_dim/2)*.01, size(X2,2), 1);
% V = randn(size(X2,2),V_dim) * .01;

max_m = 5;

lw = .1;
lV = .01;

c1 = 1e-4;
c2 = .9;
rho = .5;

[n, p] = size(X);
w = zeros(p,1);

s = [];
y = [];

[objv, gw, gV] = fm_loss(Y, X, w, X2, V);
gw = gw + lw * w;
gV = gV + lV * V;
g = [gw(:); gV(:)];

g'*g
%%
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
  alpha = 1;
  gp = g'*p;
  fprintf('epoch %d, objv %f, gp %f\n', k, objv, gp);
  for j = 1 : 10
    dw = p(1:length(w));
    dV = reshape(p((length(w)+1):end), [], V_dim);
    [new_o, gw, gV] = fm_loss(Y, X, w+alpha*dw, X2, V+alpha*dV);
    gw = gw + lw * w;
    gV = gV + lV * V;
    new_g = [gw(:); gV(:)];
    new_gp = new_g' * p;
    fprintf('alpha %f, new_objv %f, new_gp %f\n', alpha, new_o, new_gp);
    if (new_o <= objv + c1 * alpha * gp) && (new_gp >= c2 * gp)
      break;
    end
    alpha = alpha * rho;
  end

  if m == max_m
    s = s(:,2:m);
    y = y(:,2:m);
  end

  dw = p(1:length(w));
  dV = reshape(p((length(w)+1):end), [], V_dim);
  w = w + alpha * dw;
  V = V + alpha * dV;
  old_g = g;

  [objv, gw, gV] = fm_loss(Y, X, w, X2, V);
  gw = gw + lw * w;
  gV = gV + lV * V;
  g = [gw(:); gV(:)];

  s = [s, alpha*p];
  y = [y, g - old_g];
end
