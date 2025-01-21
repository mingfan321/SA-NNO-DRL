import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import math
from torch.utils.data import DataLoader
import torch
import warnings
import csv
import os
import time
import re
from train import augment
from make_dataset import make_dataset, check_extension, save_dataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ResourceWarning)

# load model
from utils import load_model
device = torch.device("cuda:0")

def angle_sort(task, depot):

    lab = task - depot
    angle = np.arctan2(lab[:, 1], lab[:, 0])
    index = angle.argsort()
    angle = angle[index]

    return angle, index

def change_node(loc, depot, MAX_LENGTHS):
    loc = np.concatenate((loc, depot[None, :]), 0)
    # distance1 = np.linalg.norm(loc[:-1] - loc[1:], axis=-1)
    # plt.scatter(loc[:, 0], loc[:, 1])
    # plt.show()
    x_max = np.max(loc[:, 0])
    x_min = np.min(loc[:, 0])
    y_max = np.max(loc[:, 1])
    y_min = np.min(loc[:, 1])
    # scale factor
    s = 1. / np.max(np.array(x_max-x_min, y_max-y_min))
    new_length = s*MAX_LENGTHS
    loc[:, 0] = s*(loc[:, 0] - x_min)
    loc[:, 1] = s*(loc[:, 1] - y_min)
    new_loc = loc[:-1]
    new_depot = loc[-1]
    # distance2 = np.linalg.norm(loc[:-1] - loc[1:], axis=-1)
    # a = distance2 / distance1
    # plt.scatter(loc[:, 0], loc[:, 1])
    # plt.show()
    # plt.close()
    return new_loc, new_depot, new_length

def cal_fitness(dataset, uav_num):
    dataloader = DataLoader(dataset, batch_size=uav_num)
    batch = next(iter(dataloader))
    N_aug = 4
    batch = augment(batch, N_aug, 'op')

    with torch.no_grad():
        collect_p, _, log_p = model(batch, return_pi=True)
    _, k_size, n_length = log_p.size()
    collect_p = collect_p.view(N_aug, uav_num, k_size).permute(1, 2, 0).reshape(-1, N_aug * k_size)
    log_p = log_p.view(N_aug, uav_num, k_size, n_length).permute(1, 2, 0, 3).reshape(-1, N_aug * k_size, n_length)
    fitness = (-collect_p).max(-1)[0]  # 适应度值
    pi = torch.gather(log_p, 1, (-collect_p).max(-1)[1].contiguous()
                      .view(uav_num, 1, 1)
                      .expand(uav_num, 1, log_p.size(-1))).squeeze(1)

    return fitness.cpu().numpy(), pi.cpu().numpy()


def sa_allocate(task, depot, prize, MAX_LENGTHS, uav_num):
    # start = time.time()
    task_num = task.shape[0]
    angle, index = angle_sort(task, depot)
    num = np.int64(np.ceil(task_num / uav_num))
    part_angle = angle[[np.int64(i * num + num / 2) for i in range(uav_num)]]  # 角度划分点

    # add depot at the end of task set
    task = np.concatenate((task, [depot]), axis=0)
    # add 0 at the end of prize set
    prize = np.append(prize, 0)

    old_index_table, ll = [], []
    for i in range(uav_num):
        a = index[i * num: (i + 1) * num] if i < uav_num - 1 else index[i * num:]
        ll.append(len(a))
        old_index_table.append(a)
    max_ll = max(ll)
    # the gap to the longest length
    d_max = max_ll - np.array(ll)
    new_index_table = old_index_table.copy()
    for i in range(uav_num):
        for j in range(d_max[i]):
            new_index_table[i] = np.append(new_index_table[i], task_num)

    dataset = []
    for i in range(uav_num):
        loc = task[new_index_table[i]]
        prz = prize[new_index_table[i]]
        dataset.append({
            'loc': torch.FloatTensor(loc).to(device),
            'prize': torch.FloatTensor(prz).to(device),
            'depot': torch.FloatTensor(depot).to(device),
            'max_length': torch.tensor(MAX_LENGTHS, dtype=torch.float32).to(device)
        })
    fitness, pi = cal_fitness(dataset, uav_num)
    # print(fitness)

    fit = sum(fitness)
    cur_fit = fit
    # transform pi into node location sequence
    pi = list(pi)
    line = []
    for i in range(uav_num):
        line.append(pi[i][(pi[i] != 0) & (pi[i] <= ll[i])] - 1)

    # return fit, new_index_table, line

    # =================SA=====================
    T_max = 80  # max_temp
    L = 4  # Markov chain length
    T_min = 60  # low_temp
    alpha = 0.9  # decay rate
    T = T_max
    best_fit = fit
    best_index = old_index_table
    best_line = line
    # plot_picture(depot, task, best_index, best_line)

    while T > T_min:
        for i in range(L):
            # produce new solution
            # obtain the uncovered node
            # node_set, prz_set = [], []  # uncovered node set, uncovered profit set
            # uncovered node index set
            unable_table = []
            for j in range(uav_num):
                unable_index = np.delete(old_index_table[j], line[j], axis=0)
                # divide the uncovered nodes into two parts
                angle, angle_index = angle_sort(task[unable_index], depot)
                # print(np.int64(unable_index.shape[0] / 2))
                # part_angle = angle[angle_index[np.int64(unable_index.shape[0]/2)]]

                unable_table.append(unable_index[angle < part_angle[j]])
                unable_table.append(unable_index[~(angle < part_angle[j])])

            # reassign nodes
            # i is odd，assign uncovered nodes by clockwise
            # new_index_table, fit_table, new_line = [], [], []  # new assignment scheme
            if i % 2 == 0:
                new_index_table, ll = [], []
                for j in range(uav_num):
                    # cat the covered node set with the assigned uncovered nodes
                    # assign the uncovered node set of first UAV to the last UAV
                    a = np.concatenate((old_index_table[j][line[j]], np.concatenate(
                        (unable_table[2 * (j + 1) - 1],
                         unable_table[2 * (j + 1)] if j < uav_num - 1 else unable_table[0]), 0)), 0)
                    ll.append(len(a))
                    new_index_table.append(a)
                max_ll = max(ll)

                d_max = max_ll - np.array(ll)
                renew_index_table = new_index_table.copy()
                for j in range(uav_num):
                    for _ in range(d_max[j]):
                        renew_index_table[j] = np.append(renew_index_table[j], task_num)

                dataset = []
                for j in range(uav_num):
                    loc = task[renew_index_table[j]]
                    prz = prize[renew_index_table[j]]
                    dataset.append({
                        'loc': torch.FloatTensor(loc).to(device),
                        'prize': torch.FloatTensor(prz).to(device),
                        'depot': torch.FloatTensor(depot).to(device),
                        'max_length': torch.tensor(MAX_LENGTHS, dtype=torch.float32).to(device)
                    })
                fit, pi = cal_fitness(dataset, uav_num)
                new_fit = sum(fit)

                pi = list(pi)
                new_line = []
                for j in range(uav_num):
                    new_line.append(pi[j][(pi[j] != 0) & (pi[j] <= ll[j])] - 1)

            else:
                new_index_table, ll = [], []
                for j in range(uav_num):
                    a = np.concatenate((old_index_table[j][line[j]], np.concatenate(
                        (unable_table[2 * j - 1] if j > 0 else unable_table[-1], unable_table[2 * j]), 0)), 0)
                    ll.append(len(a))
                    new_index_table.append(a)

                max_ll = max(ll)

                d_max = max_ll - np.array(ll)
                renew_index_table = new_index_table.copy()
                for j in range(uav_num):
                    for _ in range(d_max[j]):
                        renew_index_table[j] = np.append(renew_index_table[j], task_num)

                dataset = []
                for j in range(uav_num):
                    loc = task[renew_index_table[j]]
                    prz = prize[renew_index_table[j]]
                    # loc, dep, length = change_node(loc, dep, length)
                    dataset.append({
                        'loc': torch.FloatTensor(loc).to(device),
                        'prize': torch.FloatTensor(prz).to(device),
                        'depot': torch.FloatTensor(depot).to(device),
                        'max_length': torch.tensor(MAX_LENGTHS, dtype=torch.float32).to(device)
                    })
                fit, pi = cal_fitness(dataset, uav_num)
                new_fit = sum(fit)

                pi = list(pi)
                new_line = []
                for j in range(uav_num):
                    new_line.append(pi[j][(pi[j] != 0) & (pi[j] <= ll[j])] - 1)
                # print(new_line)
            # new_fit = sum(fit_table)
            # print(best_fit)
            # print(old_index_table)
            if new_fit > cur_fit:
                cur_fit = new_fit
                old_index_table = new_index_table
                line = new_line
                if new_fit > best_fit:
                    best_fit = new_fit
                    best_index = new_index_table
                    best_line = new_line
            else:
                if np.random.random() < math.exp((new_fit - cur_fit) / T):
                    cur_fit = new_fit
                    old_index_table = new_index_table
                    line = new_line
        T *= alpha
        # duration = time.time() - start
    return best_fit, best_index, best_line


def plot_picture(depot, task, index, line, best_fit):
    color = plt.get_cmap('Paired')

    plt.figure(figsize=(5.5, 5))
    for i in range(len(line)):
        route = task[index[i][line[i]]]
        route = np.concatenate((depot[None, :], route, depot[None, :]), 0)
        xs = route[:, 0]
        ys = route[:, 1]
        dx = np.roll(xs, -1) - xs
        dy = np.roll(ys, -1) - ys
        plt.quiver(xs, ys, dx, dy, scale_units='xy', angles='xy', scale=1, color=color.colors[2*i+1])

    plt.scatter(task[:, 0], task[:, 1], s=10)
    plt.scatter(depot[0], depot[1], marker='s', c='black', s=30)
    plt.axis([0, 1, 0, 1])
    plt.title('Total profits:{}'.format(best_fit))

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.savefig('sa_drl1.eps', dpi=1000)
    plt.show()


if __name__ == "__main__":
    model, _ = load_model('outputs/T200U5const/epoch-9.pt')
    model.to(device)
    model.eval()
    model.set_decode_type('greedy')

    fileNameList = ['op_data/unif/op_unif300_len2_seed12345.pkl',
                    'op_data/unif/op_unif400_len2_seed12345.pkl',
                    'op_data/unif/op_unif500_len2_seed12345.pkl',
                    'op_data/unif/op_unif600_len2_seed12345.pkl',
                    'op_data/unif/op_unif700_len2_seed12345.pkl',
                    'op_data/unif/op_unif800_len2_seed12345.pkl',
                    'op_data/unif/op_unif900_len2_seed12345.pkl',
                    'op_data/unif/op_unif1000_len2_seed12345.pkl']
    for filename in fileNameList:
    # filename = 'small_data/op/op_unif200_len2_seed12345.pkl'
        print(filename)
        rules = re.compile(r'op_data/unif/(.*?).pkl')
        new_filename = re.findall(rules, str(filename))[0]
        #filename = os.path.join('data/op/const', filename)
        num_samples = 30
        # np.random.seed(1234567)
        # task_num = 200
        uav_num = 5
        print('UAV num:', uav_num)
        # depot = np.array([0.5, 0.5])
        # MAX_LENGTHS = 3.
        # dim = 2
        # task = np.random.uniform(0, 1, (task_num, dim))
        # prize = torch.ones(task_num)
        data = make_dataset(filename, num_samples)
        time_table, best_fit_table, best_index_table, best_line_table = [], [], [], []
        for index, batch in enumerate(data):
            start = time.time()
            task = batch['loc']
            depot = batch['depot'].squeeze(0)
            prize = batch['prize']
            MAX_LENGTHS = batch['max_length']
            best_fit, best_index, best_line = sa_allocate(task, depot, prize, MAX_LENGTHS, uav_num)
            duration = time.time() - start
            time_table.append(duration)
            best_fit_table.append(best_fit)
            best_index_table.append(best_index)
            best_line_table.append(best_line)
            print(best_fit)
                # plot_picture(depot, task, best_index, best_line)
            # write time_rtable and best_fit_table to result file
        datadir = os.path.join('results', 'drl_small_data')
        os.makedirs(datadir, exist_ok=True)
        filename2 = os.path.join(datadir, new_filename)

        # assert not os.path.isfile(check_extension(filename2)), \
        #     "File already exists! Try running with -f option to overwrite."
        # result = list(zip(
        #     time_table,
        #     best_fit_table
        #     ))
        print("fitness:",np.array(best_fit_table).mean())
        print(np.array(time_table).mean())
        # save_dataset(result, filename2)



