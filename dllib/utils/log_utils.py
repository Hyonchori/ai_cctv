
import csv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_name = "/home/daton/PycharmProjects/pythonProject/ai_cctv/weights/classifier/mnist5_log.csv"
    with open(file_name) as f:
        reader = csv.reader(f)
        logs = np.array(list(reader)).astype(np.float32)
        print(logs)

        plt.subplot(2, 1, 1)
        plt.plot(logs[..., 0])
        plt.plot(logs[..., 2])

        plt.subplot(2, 1, 2)
        plt.plot(logs[..., 1])
        plt.plot(logs[..., 3])

        plt.show()
