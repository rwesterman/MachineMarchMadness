import re


def split_file(r_path, w_path):

    with open(r_path, 'r') as f:
        data = f.readlines()

    with open(w_path, 'w') as f:

        for line in data:
            # Default .split() merges consecutive spaces as delimiter
            line = line.split()
            for element in line:
                f.writelines("{},".format(element))
            f.writelines("\n")


    # print(data)

if __name__ == '__main__':
    split_file(".\\Training_Data\\smn_rough.txt", ".\\Training_Data\\sm_ranking.csv")

# rk1 = " 1 VILLANOVA                                 36   4   0    74.92    91.44"
#
# rk1_splt = rk1.split()
#
# print(rk1_splt)