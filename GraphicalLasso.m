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
            %计算传入lasso函数的矩阵
            jminus = setdiff(1:p,j);
            [V,D] = eig(W(jminus,jminus));
            d = diag(D);
            X = V * diag(sqrt(d)) * V'; % W_11^(1/2)
            Y = V * diag(1./sqrt(d)) * V' * S(jminus,j);    % W_11^(-1/2) * s_12

            opts = struct();
            opts.verbose = 0;

            [beta,out] = ADMM_Lasso(zeros(p-1,1),X,Y,rho,opts);

            %B(jminus,j) = beta;
            W(jminus, j) = W(jminus,jminus) * beta;
            W(j,jminus) = W(jminus,j)';
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
    % if iter == max_iter
    %     fprintf('Max iteration reached\n');
    % end
    % for j = 1:p
    %     %disp(W(jminus,jminus)\W(jminus,j) - B(jminus,j));
    %     jminus = setdiff(1:p,j);
    %     Theta(j,j) = 1 / (W(j,j) - W(j,jminus)*B(jminus,j));
    %     Theta(jminus,j) = -Theta(j,j) * B(jminus,j);
    % end
    Theta = inv(W);
end

function avg_abs_change = calculateAverageAbsoluteChange(matrix1, matrix2,p)
    avg_abs_change = norm(matrix1-matrix2,1) / (p*p);
end

%传统坐标下降法（已弃用）
function soft_threshold = softThreshold(x, rho)
    soft_threshold = sign(x) * max(abs(x) - rho, 0);
end
function beta = lasso(V,u,rho,lasso_max_iter,tol)
    n = size(u);
    beta = zeros([n,1]);
    beta_old = beta;
    lasso_it = 0;
    while lasso_it < lasso_max_iter
        for lasso_j = 1:n
            lasso_it = lasso_it + 1;
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
    if lasso_it == lasso_max_iter
        fprintf('Beta exceeded\n');
    end
end

