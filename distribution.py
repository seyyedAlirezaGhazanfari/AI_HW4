import matplotlib.pyplot as plt
import numpy as np
import os

SIZE = 100000


def dist1_dens(x):
    return 0.3 * (1 / (np.sqrt(4 * np.pi))) * np.exp((-(x - 4) ** 2) / 4) + 0.3 * (
            1 / (np.sqrt(2 * 2 * np.pi))) * np.exp((-(x - 3) ** 2) / (2 * 2)) + 0.4 * (0.01 * np.exp(-0.01 * x))


def dist2_dens(x):
    return 0.2 * (1 / np.sqrt(10 * 2 * np.pi)) * np.exp((-(x) ** 2) / (2 * 10)) + 0.2 * (
            1 / np.sqrt(15 * 2 * np.pi)) * np.exp((-(x - 20) ** 2) / (2 * 15)) + 0.3 * (
                   1 / np.sqrt(8 * 2 * np.pi)) * np.exp((-(x + 10) ** 2) / (2 * 8)) + 0.3 * (
                   1 / np.sqrt(25 * 2 * np.pi)) * np.exp((-(x - 50) ** 2) / (2 * 25))


def dist3_dens(x):
    return 0.2 * ((1 - 0.1) ** (x - 1)) * 0.1 + 0.2 * ((1 - 0.5) ** (x - 1)) * 0.5 + 0.2 * (
            (1 - 0.3) ** (x - 1)) * 0.3 + 0.4 * ((1 - 0.04) ** (x - 1)) * 0.04


def draw_original_dists():
    x = np.linspace(0, 350, SIZE)
    plt.plot(x, dist1_dens(x), color="0")
    plt.savefig("./part1/pdf1.png")
    plt.close()
    x = np.linspace(7.5, 25, SIZE)
    plt.plot(x, dist2_dens(x), color="b")
    plt.savefig("./part1/pdf2.png")
    plt.close()
    x = np.linspace(0, 70, SIZE)
    plt.plot(x, dist3_dens(x), color="r")
    plt.savefig("./part1/pdf3.png")
    plt.close()


def gaussian_sampling(mean, variance):
    # box muller algorithm
    u1, u2 = np.random.uniform(0, 1, 2)
    angel = 2 * np.pi * u2
    fixed = np.sqrt(-2 * np.log(u1))
    z1, z2 = fixed * np.cos(angel), fixed * np.sin(angel)
    return z1 * np.sqrt(variance) + mean


def exponential_sampling(lam):
    u = np.random.uniform(0, 1, 1)
    f_invert = -(1 / lam) * (np.log(1 - u))
    return f_invert


def create_custom_dists():
    list1, list2, list3 = list(), list(), list()
    for _ in range(SIZE):
        dist1 = 0.3 * gaussian_sampling(4, 2) + 0.3 * gaussian_sampling(3, 2) + 0.4 * exponential_sampling(0.01)
        dist2 = 0.2 * gaussian_sampling(0, 10) + 0.2 * gaussian_sampling(20, 15) + 0.3 * gaussian_sampling(-10,8) + 0.3 * gaussian_sampling(50, 25)
        dist3 = 0.2 * geometric_sampling(0.1) + 0.2 * geometric_sampling(0.5) + 0.2 * geometric_sampling(0.3) + 0.4 * geometric_sampling(0.04)
        dist3 = float(dist3)
        dist1 = float(dist1)
        dist2 = float(dist2)
        list1.append(dist1)
        list2.append(dist2)
        list3.append(dist3)
    create_log_file(list1, list2, list3)
    plt.hist(list1, color="0")
    plt.savefig("./part1/pdf1_sample.png")
    plt.close()
    plt.hist(list2, color="b")
    plt.savefig("./part1/pdf2_sample.png")
    plt.close()
    plt.hist(list3, color="r")
    plt.savefig("./part1/pdf3_sample.png")
    plt.close()


def create_log_file(list1, list2, list3):
    dist1_mean = np.mean(list1)
    dist2_mean = np.mean(list2)
    dist3_mean = np.mean(list3)
    dist1_var = np.std(list1)
    dist2_var = np.std(list2)
    dist3_var = np.std(list3)
    with open("./part1/log.txt", "w") as file:
        file.flush()
        file.write("{:.4f} {:.4f}\n".format(dist1_mean, dist1_var))
        file.write("{:.4f} {:.4f}\n".format(dist2_mean, dist2_var))
        file.write("{:.4f} {:.4f}\n".format(dist3_mean, dist3_var))
        file.close()


def geometric_sampling(p):
    u = np.random.uniform(0, 1, 1)
    q = 1 - p
    return np.floor(np.log(u) / np.log(q)) + 1


def run():
    os.makedirs("part1")
    draw_original_dists()
    create_custom_dists()


if __name__ == '__main__':
    run()
