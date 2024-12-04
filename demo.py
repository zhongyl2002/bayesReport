import numpy as np
from scipy.stats import dirichlet, multinomial
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体的字体名称
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 贝叶斯推断过程的函数
def bayesian_inference(n:int, observed_data:list, alpha_prior:list):
    """
    贝叶斯推断函数：使用狄利克雷先验，计算后验分布的参数。
    
    参数：
    - n (int)：样本总数（实验次数）。
    - observed_data (list)：每个类别观察到的次数，例如 [x1, x2, x3, x4, x5]。
    - alpha_prior (list)：先验分布的超参数，例如 [1, 1, 1, 1, 1]。
    
    返回：
    - posterior_alpha (list)：后验分布的超参数。
    - posterior_sample (ndarray)：从后验分布中生成的样本。
    """
    # 1. 计算后验分布的超参数
    posterior_alpha = [alpha + x for alpha, x in zip(alpha_prior, observed_data)]
    
    # 2. 使用狄利克雷分布生成后验分布样本
    posterior_dist = dirichlet(posterior_alpha)
    posterior_sample = posterior_dist.rvs(size=1)
    
    return posterior_alpha, posterior_sample


# 绘制多项式分布函数图像
def visual(posterior_sample:list, iter:int):
    # 可视化后验分布样本
    plt.figure(figsize=(8, 6))
    plt.bar(range(5), posterior_sample[0], tick_label=["类1", "类2", "类3", "类4", "类5"])
    plt.xlabel("类别")
    plt.ylabel("后验概率")
    plt.title("后验概率（iter=%d)"%(iter))
    plt.show()

# 各类别观察到的次数
observed_data = [   [10, 30, 10, 10, 40],
                    [5, 35, 10, 10, 40],
                    [10, 30, 10, 15, 35],
                    [5, 25, 10, 10, 50],
                    [10, 40, 10, 5, 35]]
# for i in range(len(observed_data)):
#     print(sum(observed_data[i]))

prior_alpha = [1, 1, 1, 1, 1]   # 无信息的狄利克雷先验（均匀分布）
posterior_alpha = prior_alpha   # 后验分布的超参数
initial_posterior_sample = dirichlet(posterior_alpha).rvs(size=1)
print(initial_posterior_sample)
visual(initial_posterior_sample, 0)

for i in range(len(observed_data)):
    # 执行贝叶斯推断
    posterior_alpha, posterior_sample = bayesian_inference(sum(observed_data[i]), observed_data[i], prior_alpha)
    prior_alpha = posterior_alpha
    print("后验分布的超参数(iter=%d)："%(i), posterior_alpha)
    print("从后验分布中生成的样本(iter=%d)："%(i), sep=" ")
    print(posterior_sample)

    # 可视化后验分布样本
    visual(posterior_sample, i + 1)
