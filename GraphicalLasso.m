% Graphical Lasso
function [Theta,W] = GraphicalLasso(S,rho,max_iter,t)
    %初始化
    p = size(S,1);
    W = S + rho * eye(p);
    Theta = zeros(p,p);
    W_old = W;
    iter = 0;
    B = zeros(p,p); %存储beta

    %计算 eps = t*ave|S^{-diag}|
    eps = 0.01 * sum(abs(S - diag(diag(S)))) / (p*p);
    %iterate
    while iter < max_iter
        for j = 1:p
            iter = iter + 1;
            V = W([1:j-1, j+1:end], [1:j-1, j+1:end]);
            u = S([1:j-1, j+1:end], j);
            beta = lasso(V,u,rho,max_iter,t);
            B(:,j) = [beta(1:j-1); -1; beta(j:end)];
            W([1:j-1, j+1:end], j) = V * beta;
            if calculateAverageAbsoluteChange(W,W_old) < eps
                break;
            end
        end
        if calculateAverageAbsoluteChange(W,W_old) < eps
            break;
        end
    end
    %calculate Theta
    if iter == max_iter
        fprintf('Max iteration reached\n');
    end
    for j = 1:p
        Theta(j,j) = 1 / (W(j,j) - W([1:j-1, j+1:end],j)'*B([1:j-1, j+1:end],j));
        Theta(:,j) = -Theta(j,j) * B(:,j);
    end
end

function avg_abs_change = calculateAverageAbsoluteChange(matrix1, matrix2)
    p = size(matrix1,1);
    avg_abs_change = sum(abs(matrix1-matrix2)) / (p*p);
end

function soft_threshold = softThreshold(x, rho)
    soft_threshold = sign(x) * max(abs(x) - rho, 0);
end

function beta = lasso(V,u,rho,max_iter,eps)
    max_iter = max(max_iter, 100);
    n = size(u);
    beta = zeros(n);
    beta_old = beta;
    iter = 0;
    while iter < max_iter
        for j = 1:n
            iter = iter + 1;
            %calculate sum
            sum = 0;
            for k = 1:n
                if k ~= j
                    sum = sum + V(j,k) * beta_old(k);
                end
            end
            beta(j) = softThreshold(u(j) - sum, rho) / V(j,j);
        end
        if norm(beta - beta_old) < eps
            break;
        end
        beta_old = beta;
    end
    if iter == max_iter
        fprintf('Beta exceeded\n');
    end
end

