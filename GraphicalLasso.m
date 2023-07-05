% Graphical Lasso
function [Theta,W] = GraphicalLasso(S,rho,max_iter,tol)
    %初始化
    p = size(S,1);
    W = S + rho * eye(p);
    Theta = zeros(p,p);
    W_old = W;
    iter = 0;
    B = zeros(p,p); %存储beta

    %计算 eps = t*ave|S^{-diag}|
    eps = 0.001 * norm(S - diag(diag(S)),1) / (p * (p-1));
    %iterate
    while iter < max_iter
        for j = 1:p
            iter = iter + 1;
            jminus = setdiff(1:p,j);
            V = W(jminus,jminus);
            u = S(jminus, j);
            beta = lasso(V,u,rho,max_iter,tol);
            B(jminus,j) = beta;
            W(jminus, j) = V * beta;
            W(j,jminus) = V * beta;
            if calculateAverageAbsoluteChange(W, W_old, p) < eps
                break;
            end
            W_old = W;
        end
        if calculateAverageAbsoluteChange(W, W_old, p) < eps
            break;
        end
    end
    %calculate Theta
    if iter == max_iter
        fprintf('Max iteration reached\n');
    end
    for j = 1:p
        jminus = setdiff(1:p,j);
        Theta(j,j) = 1 / (W(j,j) - W(j,jminus)*B(jminus,j));
        Theta(jminus,j) = -Theta(j,j) * B(jminus,j);
    end
end

function avg_abs_change = calculateAverageAbsoluteChange(matrix1, matrix2,p)
    avg_abs_change = norm(matrix1-matrix2,1) / (p*p);
end

function soft_threshold = softThreshold(x, rho)
    soft_threshold = sign(x) * max(abs(x) - rho, 0);
end

function beta = lasso(V,u,rho,max_iter,tol)
    n = size(u);
    beta = zeros([n,1]);
    beta_old = beta;
    it = 0;
    while it < max_iter
        for lasso_j = 1:n
            it = it + 1;
            %calculate sum
            sum = 0;
            for lasso_k = 1:n
                if lasso_k ~= lasso_j
                    sum = sum + V(lasso_k,lasso_j) * beta_old(lasso_k);
                end
            end
            beta(lasso_j) = softThreshold(u(lasso_j) - sum, rho) / V(lasso_j,lasso_j);
            if norm(beta - beta_old,1) < tol
                break;
            end
            beta_old = beta;
        end
        if norm(beta - beta_old,1) < tol
            break;
        end
    end
    if it == max_iter
        fprintf('Beta exceeded\n');
    end
end

