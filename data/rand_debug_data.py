import random

if __name__ == "__main__":
    with open("data/mini_train_0_1.csv", "w") as file:
        for i in range(20):
            label = [str(random.choice([0,1]))]
            label_num = int(label[0])
            size = 256
            values = [str(random.randrange(label_num*size//2, (label_num+1)*size//2)) for i in range(10) ]
            file.write(",".join(label + values) + "\n")
