function[z,u,history] = lassoADMM(AtA,AtY,yTy,n,p,lambda, rho,alpha,z,u,absTol,relTol,maxIter)
% Internal use only.

%   Copyright 2017-2019 The MathWorks, Inc

[L,U] = cholFactors(AtA,rho);
objFun = @(beta,z)lassoObjective(AtA,AtY,yTy,beta,z,lambda,n);

for k = 1: maxIter
    
    % x-update
    q = AtY + rho*(z-u);
    x = U\(L\q);
    
    % z-update
    zold = z;
    x_hat = alpha*x + (1-alpha) * zold;
    z = localShrinkage(x_hat + u, lambda*n/rho);
    
    % u-update
    u = u + (x_hat -z);
    
    % convergence check
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(p)*absTol + relTol*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(p)*absTol + relTol*norm(rho*u);
    
    % Objective
    [history.objective(k), history.mse(k)] = objFun(x,z);      
    
    % Convergence
    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
    if k == maxIter
        warning(message('stats:lasso:MaxIterReached',num2str(lambda)));
    end
end

end


function z = localShrinkage(x, kappa)
    z = max(0, x-kappa) - max(0, -x-kappa );
end

function [L, U] = cholFactors(xtx, rho)
L = chol(xtx+rho*eye(size(xtx,1)), 'lower');
U = L';
end

function [f, mse] = lassoObjective(xtx,xty,yty,b,z,lambda,n)
mse = (b'*xtx*b - 2*b'*xty + yty)/n;
f = mse/2 + lambda*sum(abs(z));
end
