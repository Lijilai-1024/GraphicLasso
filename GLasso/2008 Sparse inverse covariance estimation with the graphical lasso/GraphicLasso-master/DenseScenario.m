% 测试GraphicalLasso函数
% 初始参数
p = 3; % 变量个数
n = 1000; % 样本个数
max_iter = 100000; % 最大迭代次数
tol = 0.0001; % 收敛阈值
rho = 0.01; % 惩罚系数

% 定义逆协方差矩阵
inv_cov_matrix = ones(p,p);
inv_cov_matrix = inv_cov_matrix + diag(ones(p,1));
disp(inv_cov_matrix);
mu = zeros(p,1);
data = mvnrnd(mu,inv(inv_cov_matrix),n);


S = cov(data);% 样本协方差矩阵

tic;
[Theta, W] = GraphicalLasso(S,rho,max_iter,tol);
time = toc;
disp(['GLasso运行时间：', num2str(time), '秒']);

tic;
[Theta1, W1] = StandardGraphicalLasso(S,rho,max_iter,tol);
time = toc;
disp(['StandardGLasso运行时间：',num2str(time),'秒']);

