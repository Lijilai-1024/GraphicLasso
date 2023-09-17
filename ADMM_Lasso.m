function [x,out] = ADMM_Lasso(x0,A,b,mu,opts)
    %Init Step
    %最大迭代次数
    if ~isfield(opts, 'maxit'); opts.maxit = 5000; end 
    %增广拉格朗日系数
    if ~isfield(opts, 'sigma'); opts.sigma = 0.01; end
    %停机条件
    if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
    if ~isfield(opts, 'gtol'); opts.gtol = 1e-14; end
    %更新步长
    if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
    %verbose=1输出迭代过程
    if ~isfield(opts, 'verbose'); opts.verbose = 1; end
    k = 0;
    tt = tic;
    x = x0;
    out = struct();
    %初始化 ADMM 的辅助变量 y, z，其维度均与 x 相同。
    [~,n] = size(A);
    sm = opts.sigma;
    y = zeros(n,1);
    z = zeros(n,1);
    %计算并记录起始点的目标函数值。
    fp = inf; nrmC = inf;%存储目标函数和自变量的变化量
    f = Func(A, b, mu, x);
    f0 = f;
    out.fvec = f0;
    %预存储变量用于加速迭代
    AtA = A' * A;
    R = chol(AtA + opts.sigma*eye(n));
    Atb = A'*b;

    while k < opts.maxit && abs(f - fp) > opts.ftol && nrmC > opts.gtol%停机条件
        fp = f;

        %通过求偏导，令偏导数为0更新x
        x = R \ (R' \ (Atb + sm*z - y));

        %用传统坐标下降法更新z
        c = x + y/sm;
        z = prox(c, mu/sm);
        
        %梯度下降法更新y
        y = y + opts.gamma * sm * (x - z);

        f = Func(A, b, mu, x);
        nrmC = norm(x - z, 2);
        if opts.verbose
            fprintf('itr: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
        end
        k = k + 1;
        out.fvec = [out.fvec; f];
    end
    %存储输出值
    out.y = y;
    out.fval = f;
    out.itr = k;
    out.tt = toc(tt);
    out.nrmC = norm(c - y, inf);
end
%软阈值操作
function y = prox(x, mu)
    y = max(abs(x) - mu, 0);
    y = sign(x) .* y;
end

%Lasso问题的目标函数
function f = Func(A, b, mu, x)
    w = A * x - b;
    f = 0.5 * (w' * w) + mu*norm(x,1);
end
