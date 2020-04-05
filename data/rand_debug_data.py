import random

if __name__ == "__main__":
    with open("data/mini_train_0_1.csv", "w") as file:
        for i in range(100):
            label = [str(random.choice([0,1]))]
            label_num = int(label[0])
            values = list()
            if label[0] == "0":
                for j in [0, 0]:
                    values.append(str(j))
            else:
                for j in [255, 255]:
                    values.append(str(j))
            file.write(",".join(label + values) + "\n")
