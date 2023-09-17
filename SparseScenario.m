% 测试GraphicalLasso函数
% 初始参数
p = 500; % 变量个数
n = 10000; % 样本个数
max_iter = 1000; % 最大迭代次数
tol = 0.0001; % 收敛阈值
rho = 0.01; % 惩罚系数

% 逆协方差矩阵
inv_cov_matrix = zeros(p);
for i = 1:p-1
    inv_cov_matrix(i,i) = 1;
    inv_cov_matrix(i,i+1) = 0.5;
    inv_cov_matrix(i+1,i) = 0.5;
end
inv_cov_matrix(p,p) = 1;

%生成多维正态分布数据
mu = zeros(p,1);
data = mvnrnd(mu,inv(inv_cov_matrix),n);


S = cov(data);% 样本协方差矩阵
[Theta, W] = GraphicalLasso(S,rho,max_iter,tol);
%[Theta1, W1] = StandardGraphicalLasso(S,rho,max_iter,tol);
disp(Theta);
