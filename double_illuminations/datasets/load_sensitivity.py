import numpy as np
import matplotlib.pyplot as plt

#  400nm to 720nm in an interval of 10nm
# three rows represent red, green and blue channel


datapath = "datasets/camspec_database.txt"

Pixel_txt = "datasets/Pixel4.txt"


def load():
    sensitivity_list = []

    with open(datapath, 'r', encoding="UTF-8") as source:
        lines = source.readlines()
        num = len(lines) // 4

        for i in range(num):
            r_list = [float(x) for x in lines[4 * i + 1].strip().split()]
            g_list = [float(x) for x in lines[4 * i + 2].strip().split()]
            b_list = [float(x) for x in lines[4 * i + 3].strip().split()]

            sensitivity_list.append([r_list, g_list, b_list])

            # plt.plot(range(400, 730, 10), b_list, 'b-', range(400, 730, 10), g_list, 'g-',
            #          range(400, 730, 10), r_list, 'r-')
            # plt.legend(['b', 'g', 'r'], loc='best')
            # plt.title(str(i))
            # plt.show()

    return sensitivity_list


def load_Pixel():
    sensitivity_list = [[], [], []]

    with open(Pixel_txt, 'r', encoding="UTF-8") as source:
        lines = source.readlines()

        for line in lines:
            line = line.split("	")
            sensitivity_list[0].append(float(line[0].strip()))
            sensitivity_list[1].append(float(line[1].strip()))
            sensitivity_list[2].append(float(line[2].strip()))

        # plt.plot(range(400, 710, 10), sensitivity_list[2], 'b-', range(400, 710, 10), sensitivity_list[1], 'g-',
        #          range(400, 710, 10), sensitivity_list[0], 'r-')
        #
        # plt.legend(['b', 'g', 'r'], loc='best')
        # plt.title("Pixel")
        # plt.show()

    return sensitivity_list


# sensitivity_list = np.array(load())[:-3]
#
# while True:
#     weights = np.array([np.random.rand() for _ in range(28)])
#
#     weights = np.zeros(25)
#
#     a = np.arange(0, 25)
#     np.random.shuffle(a)
#
#     weights[a[:np.random.randint(1, 3)]] = 1
#
#     weights = weights * np.random.random(25)
#
#
#     # weights[np.random.randint(0, 28)] = 1
#     # weights[np.random.randint(0, 28)] = 1
#
#     weights = weights / np.sum(weights)
#
#     sens = np.einsum("i,ijk->jk", weights, sensitivity_list)
#
#     plt.plot(range(400, 730, 10), sens[2], 'b-', range(400, 730, 10), sens[1], 'g-', range(400, 730, 10), sens[0], 'r-')
#     plt.legend(['b', 'g', 'r'], loc='best')
#     plt.title("curve")
#     plt.show()
