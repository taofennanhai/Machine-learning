set=[0]*5
total=0
with open("./data/SST2/dev.txt") as f:
    lines=f.readlines()
    for line in lines:
        total+=1
        if line=="\n":
            continue
        # print(line)
        score=line.split(" ")[0]
        set[int(score)]+=1
for i in range(5):
    print("dev percent of class i:",set[i]/total)

set=[0]*5
total=0
with open("./data/SST2/test.txt") as f:
    lines=f.readlines()
    for line in lines:
        total+=1
        if line=="\n":
            continue
        # print(line)
        score=line.split(" ")[0]
        set[int(score)]+=1
for i in range(5):
    print("test percent of class i:",set[i]/total)