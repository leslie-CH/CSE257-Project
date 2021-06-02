import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style({"axes.facecolor": ".95"})


def arrow(value):
    a = None
    if value == 0:
        a = r'$\leftarrow$'
    elif value == 1:
        a = r'$\uparrow$'
    elif value == 2:
        a = r'$\rightarrow$'
    elif value == 3:
        a = r'$\downarrow$'
    else:
        ValueError('Arrow value error!')
    return a


def plot_matrix(value, save_name, policy=False):
    value = value.reshape(9, 9)
    # np.set_printoptions(precision=4)
    # print(value)
    plt.figure(figsize=(16, 12))
    plt.imshow(value, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    fontsize = 24

    # Loop over data dimensions and create text annotations.
    for i in range(9):
        for j in range(9):
            if policy:
                plt.text(j, i, arrow(value[i, j]), ha="center", va="center", color="w", fontsize=fontsize+14)
            else:
                plt.text(j, i, "%0.1f" %value[i, j], ha="center", va="center", color="w", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


transition = np.zeros((4, 81, 81))      # a * s * s'
for i in range(4):
    f = open('prob_a' + str(i+1) + '.txt', "r")
    lines = f.readlines()
    for line in lines:
        data = line.split()
        transition[i, int(data[0])-1, int(data[1])-1] = float(data[-1])
    f.close()


f = open('rewards.txt', "r")
lines = f.readlines()
reward = []
for line in lines:
    reward.append([float(i) for i in line.split()])
f.close()
reward = np.array(reward)   # R(s) only depends on s





seed = 0
n = 81
gama = 0.99

# policy iteration
np.random.seed(seed=seed)
policy = np.random.randint(4, size=81)
policy = np.vstack((policy, np.arange(81)))     # policy: a * s

# Policy evaluation
value = np.linalg.inv(np.identity(n) - gama * transition[policy[0], policy[1], :]) @ reward

for t in range(500):
    value_old = value.copy()

    # Policy improvement
    policy[0] = np.argmax((transition * value.flatten()).sum(axis=-1), axis=0)

    # Policy evaluation
    value = np.linalg.inv(np.identity(n) - gama * transition[policy[0], policy[1], :]) @ reward

    if (value == value_old).all():
        print('iterations:', t)
        break

plot_matrix(value.reshape(9, 9).T, 'policy_iteration_value.png')
# print(policy[0].reshape(9, 9).T)
plot_matrix(policy[0].reshape(9, 9).T, 'policy_iteration_policy.png', policy=True)


# Value iteration
value_v = np.zeros(81, dtype=int)
# gama = 1.

for t in range(2000):
    value_old = value_v.copy()

    value_v = reward.flatten() + gama * np.max((transition * value_v).sum(axis=-1), axis=0)

    if np.abs(value_v - value_old).sum() < 1e-5:
        print('iterations:', t)
        break

plot_matrix(value_v.reshape(9, 9).T, 'value_iteration_value.png')
# print(policy[0].reshape(9, 9).T)
plot_matrix(policy[0].reshape(9, 9).T, 'value_iteration_policy.png', policy=True)




# # algorithm in the paper
# K = 10
# H = 100
# Q = np.zeros(shape=(K, H+1, 81, 4))
# V = np.zeros(shape=(K, H+1, 81))
# D = []
#
# np.random.seed(seed=seed)
# policy = np.random.randint(4, size=(H, 81))     # random policy at beginning
# S = np.arange(81)
# # policy = np.vstack((policy, S))     # policy: a * s
# v_alg = []
#
# for k in range(K):
#     s_1 = np.random.randint(81)  # choose s_1 randomly, uniform distribution
#
#     if k > 0:
#         for h in range(H-1, -1, -1):
#             f_h = np.zeros(shape=(81, 4, 2))    # s * a, count
#             V[k, h+1] = np.max(Q[k, h+1], axis=-1)
#
#             for D_h in D:
#                 # for i, (s_h, a_h, r_h) in enumerate(D_h[:-1]):
#                 #     # # f_h[s_h, a_h, 0] += r_h + np.max(Q[k, h+1, D_h[i+1][0]])
#                 #     f_h[s_h, a_h, 0] += r_h + V[k, h+1, D_h[i+1][0]]
#                 #     f_h[s_h, a_h, 1] += 1
#
#                 D_h_arr = np.array(D_h)
#                 s_plus_1 = D_h_arr[:, 0][1:].astype(int)
#                 D_h_arr = D_h_arr[:-1]
#                 s = D_h_arr[:, 0].astype(int)
#                 a = D_h_arr[:, 1].astype(int)
#                 r = D_h_arr[:, 2].astype(float)
#                 f_h[s, a, 0] += r + V[k, h+1, s_plus_1]
#                 f_h[s, a, 1] += 1
#
#             Q[k, h][f_h[:, :, 1] != 0] = f_h[f_h[:, :, 1] != 0][:, 0] / f_h[f_h[:, :, 1] != 0][:, 1]
#             policy[h] = np.argmax(Q[k, h], axis=-1)
#
#     V[k, 0] = np.max(Q[k, 0], axis=-1)
#     v_alg.append(V[k, 0].sum())
#
#     s_h = s_1
#     D_k = []
#     for h in range(H):
#         a_h = policy[h, s_h]
#         r_h = reward[s_h, 0]
#         D_k.append([s_h, a_h, r_h])
#         s_h = np.random.choice(S, p=transition[a_h, s_h, :])
#         if h == H-1:
#             D_k.append([s_h, None, None])
#
#     D.append(D_k)
#
#
# plot_matrix(V[-1, 0].reshape(9, 9).T, 'alg_value.png')
# print(policy[0].reshape(9, 9).T)
# plot_matrix(policy[0].reshape(9, 9).T, 'alg_policy.png', policy=True)
#
# sns.set_style("darkgrid", {"axes.facecolor": ".95"})
# plt.plot(list(range(K)), v_alg)
# plt.savefig('alg_value_sum.png')
# plt.show()



import os
import torch
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # change GPU here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

K = 2000
H = 100
save_path = str(K) + '_' + str(H) + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

Q = torch.zeros(size=(K, H+1, 81, 4), device=device)
V = torch.zeros(size=(K, H+1, 81), device=device)
D = []

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

policy = torch.randint(4, size=(H, 81))     # random policy at beginning
S = torch.arange(81, device=device)
reward = torch.from_numpy(reward).to(device)
v_alg = []

for k in tqdm(range(K)):
    s_1 = torch.randint(81, (1,))  # choose s_1 randomly, uniform distribution

    if k > 0:
        for h in range(H-1, -1, -1):
            f_h = torch.zeros(size=(81, 4, 2), device=device)    # s * a, count
            V[k, h+1], max_indices = torch.max(Q[k, h+1], dim=-1)

            for D_h in D:
                s_plus_1 = D_h[:, 0][1:].to(int)
                D_h = D_h[:-1]
                s = D_h[:, 0].to(int)
                a = D_h[:, 1].to(int)
                r = D_h[:, 2].to(float)
                f_h[s, a, 0] += r + V[k, h+1, s_plus_1]
                f_h[s, a, 1] += 1

            Q[k, h][f_h[:, :, 1] != 0] = f_h[f_h[:, :, 1] != 0][:, 0] / f_h[f_h[:, :, 1] != 0][:, 1]
            policy[h] = torch.argmax(Q[k, h], dim=-1)

    V[k, 0], max_indices = torch.max(Q[k, 0], dim=-1)
    v_alg.append(V[k, 0].sum().item())

    s_h = s_1
    D_k = torch.zeros(size=(H, 3), device=device)
    for i, h in enumerate(range(H)):
        a_h = policy[h, s_h]
        r_h = reward[s_h, 0]
        D_k[i] = torch.tensor([s_h, a_h, r_h])
        s_h = np.random.choice(S.cpu(), p=transition[a_h, s_h, :])
        # p = transition[a_h, s_h, :]
        # idx = p.multinomial(num_samples=1, replacement=True)
        # s_h = S[idx]

        if h == H-1:
            D_k[i] = torch.tensor([s_h, np.nan, np.nan])

    D.append(D_k)

plot_matrix(V[-1, 0].cpu().numpy().reshape(9, 9).T, save_path + '/alg_value.png')
# print(policy[0].numpy().reshape(9, 9).T)
plot_matrix(policy[0].cpu().numpy().reshape(9, 9).T, save_path + '/alg_policy.png', policy=True)

sns.set_style("darkgrid", {"axes.facecolor": ".95"})
plt.figure()
plt.plot(list(range(K)), v_alg)
plt.tight_layout()
plt.savefig(save_path + '/alg_value_sum.png')
plt.show()
