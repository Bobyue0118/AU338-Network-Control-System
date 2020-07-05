import numpy as np
import math
import matplotlib.pyplot as plt

# 算例实例化
############################################################################
# 假定效用函数为Us(x) = ln(x)
# 四条流s1，s2，s3, s4, 传输时延Ds
s1 = np.array([1, 2, 3, 4])
s2 = np.array([1, 2, 5, 6])
s3 = np.array([7, 2, 3, 4])
s4 = np.array([7, 2, 5, 6])
Ds = 10

# S(l)
s_l = [(1, 2), (1, 2, 3, 4), (1, 3), (1, 3), (2, 4), (2, 4), (3, 4)]

# 冲突域Λ
r1 = np.array([1, 2, 3, 5, 7])
r2 = np.array([2, 3, 4, 5, 6])
tmp = np.array([val for val in r1 if val in r2])  # 两个冲突域重叠部分


def comb(r1, r2, tmp):  # 所有可能的链路组合
    get1 = list(tmp)
    for i in r1:
        if i not in tmp:
            for j in r2:
                if j not in tmp:
                    get1.append((i, j))
    return get1


get = comb(r1, r2, tmp)
#############################################################################

# 初始值设定
#############################################################################
# 对偶算子迭代步长
beta = 0.00005
gamma = 0.00005

# 初始化对偶算子表，横轴时间，纵轴节点或链路
num = 15000  # 对偶算子迭代次数
lian = 7  # 链路个数
liu = 4  # 流的个数
p = np.array([5.30] * lian)  # 拉格朗日系数p初始化
m = np.array([2.10] * liu)  # 拉格朗日系数μ初始化

# 初始化各链路裕度变量σ
sigma = np.array([0.1] * lian)

# 初始化每一条链路l的最大速率c，c_l代表实际速率（有的会被置零））
c = np.array([4.0] * lian)
############################################################################

# 迭代过程用于记录的数组
k = len(get)
lian_choose = np.array([0.0] * k)
xs1 = np.array([0.0] * num)
xs2 = np.array([0.0] * num)
xs3 = np.array([0.0] * num)
xs4 = np.array([0.0] * num)
m_ = [[] for i in range(4)]
p_ = [[] for i in range(7)]

# 迭代过程
for t in range(num):
    # 选出使p_l*c_l最大的链路
    a = 0
    for i in get:
        if a <= len(tmp) - 1:
            lian_choose[a] = p[i - 1] * c[i - 1]
        else:
            lian_choose[a] = p[i[0] - 1] * c[i[0] - 1] + p[i[1] - 1] * c[i[1] - 1]
        a += 1
    index = np.argmax(lian_choose)

    # 计算x_s*
    qs1 = np.sum(np.array([p[val - 1] for val in s1]))
    qs2 = np.sum(np.array([p[val - 1] for val in s2]))
    qs3 = np.sum(np.array([p[val - 1] for val in s3]))
    qs4 = np.sum(np.array([p[val - 1] for val in s4]))

    def us_d_i(qs):  # us求导再求反函数
        x = 1 / qs
        return x
    xs1[t] = us_d_i(qs1)
    xs2[t] = us_d_i(qs2)
    xs3[t] = us_d_i(qs3)
    xs4[t] = us_d_i(qs4)

    # 计算σ_l*
    ita = np.array([0.0] * lian)
    for i in range(lian):
        if len(s_l[i]) == 2:
            ita[i] = m[s_l[i][0] - 1] + m[s_l[i][1] - 1]
        else:
            ita[i] = m[s_l[i][0] - 1] + m[s_l[i][1] - 1] + m[s_l[i][2] - 1] + m[s_l[i][3] - 1]

    for i in range(lian):
        sigma[i] = math.sqrt(ita[i] / p[i])

    # 更新p_l
    c_l = np.array([0.0] * lian)
    if type(get[index]) == tuple:
        c_l[get[index][0] - 1] = c[get[index][0] - 1]
        c_l[get[index][1] - 1] = c[get[index][1] - 1]
    else:
        c_l[get[index] - 1] = c[get[index] - 1]

    for i in range(lian):
        if len(s_l[i]) == 4:
            p[i] = max(0.01, p[i] + beta * (sigma[i] - c_l[i] + xs1[t] + xs2[t] + xs3[t] + xs4[t]))
        elif s_l[i] == (1, 2):
            p[i] = max(0.01, p[i] + beta * (sigma[i] - c_l[i] + xs1[t] + xs2[t]))
        elif s_l[i] == (1, 3):
            p[i] = max(0.01, p[i] + beta * (sigma[i] - c_l[i] + xs1[t] + xs3[t]))
        elif s_l[i] == (2, 4):
            p[i] = max(0.01, p[i] + beta * (sigma[i] - c_l[i] + xs2[t] + xs4[t]))
        elif s_l[i] == (3, 4):
            p[i] = max(0.01, p[i] + beta * (sigma[i] - c_l[i] + xs3[t] + xs4[t]))
        p_[i].append(p[i])

    # 更新m_s
    sigma1 = np.array([0.0] * lian)
    # print(sigma)
    for i in range(lian):
        sigma1[i] = 1 / sigma[i]
        # print(sigma1[i])

    m[0] = max(0, m[0] + gamma * (sigma1[0] + sigma1[1] + sigma1[2] + sigma1[3] - Ds))
    m[1] = max(0, m[1] + gamma * (sigma1[0] + sigma1[1] + sigma1[4] + sigma1[5] - Ds))
    m[2] = max(0, m[2] + gamma * (sigma1[6] + sigma1[1] + sigma1[2] + sigma1[3] - Ds))
    m[3] = max(0, m[3] + gamma * (sigma1[6] + sigma1[1] + sigma1[4] + sigma1[5] - Ds))
    m_[0].append(m[0])
    m_[1].append(m[1])
    m_[2].append(m[2])
    m_[3].append(m[3])


# 总效用函数计算
def us_calculation():
    return np.log(xs1) + np.log(xs2) + np.log(xs3) + np.log(xs4)


# 可视化部分
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(us_calculation())
plt.xlabel('迭代次数')
plt.ylabel("效用函数Us")
plt.title("效用函数值变化曲线")
plt.savefig("效用函数值变化曲线.png")
plt.show()

xs = [xs1, xs2, xs3, xs4]
xs_name = ['Xs1', 'Xs2', 'Xs3', 'Xs4']
for i in range(len(xs)):
    plt.plot(xs[i])
    plt.xlabel('迭代次数')
    plt.ylabel("信号源速率" + xs_name[i])
    plt.title("信号源速率" + xs_name[i] + "变化曲线")
    plt.savefig("信号源速率" + xs_name[i] + "变化曲线.png")
    plt.show()

p_name = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
for i in range(7):
    plt.plot(p_[i])
    plt.xlabel('迭代次数')
    plt.ylabel("拉格朗日系数" + p_name[i])
    plt.title("拉格朗日系数" + p_name[i] + "变化曲线")
    plt.savefig("拉格朗日系数" + p_name[i] + "变化曲线.png")
    plt.show()

m_name = ['m1', 'm2', 'm3', 'm4']
for i in range(4):
    plt.plot(m_[i])
    plt.xlabel('迭代次数')
    plt.ylabel("拉格朗日系数" + m_name[i])
    plt.title("拉格朗日系数" + m_name[i] + "变化曲线")
    plt.savefig("拉格朗日系数" + m_name[i] + "变化曲线.png")
    plt.show()
