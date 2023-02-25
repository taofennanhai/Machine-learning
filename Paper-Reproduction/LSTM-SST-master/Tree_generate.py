files = ["train.txt", "dev.txt",'test.txt']


def dp(info: list, out):
    st = []
    comp = []
    info = list(info)
    if info[0] == ' ':
        info.pop(0)
    for i in range(len(info)):
        if info[i] == '(':
            st.append(i)
            comp.append(-1)
        elif info[i] == ')':
            for j in range(len(comp) - 1, -1, -1):
                if comp[j] == -1:
                    comp[j] = i
                    break
    score = int(info[st[0] + 1])
    if len(st)==1:
        word=''.join(info[st[0]+2:comp[0]])

    else:
        if comp[1]!=len(info)-1:#如果还存在其他元素
            word1=dp(info[st[1]:comp[1]+1],out)
            word2=dp(info[comp[1]+1:],out)
            print(word1)
            print(word2)
            word=word1+' '+word2
    out.write("{} {}\n".format(score,word))
    return word


for file in files:
    with open("./data/SST/" + file) as f:
        with open("./data/Tree/" + file,'w') as out:
            lines = f.readline()
            while lines:
                dp(list(lines), out)
                lines=f.readline()
