import torch


class Reader(torch.utils.data.Dataset):
    def __init__(self, file_path,max_len):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            max_line = 0
            line_num = 0
            for line in lines:
                # self.score.append(int(line.split(' ')[0]))
                words = line.split(' ')[1:]
                while '' in words:
                    words.remove('')
                words = [int(i) for i in words]
                if len(words) > max_line:
                    max_line = len(words)
                line_num += 1
            self.line_num = line_num
            self.max_line = max(max_len, max_line)
        self.files = open(file_path, 'r', encoding='utf-8')
        self.score=[]
        self.sentence=[]
        lines = self.files.readlines()
        for line in lines:
            words = line.split(' ')
            score = int(words[0])
            while '' in words:
                words.remove('')
            sentence = [int(i) for i in words[1:]]
            while len(sentence)<self.max_line:
                sentence.append(0)
            # score = torch.tensor(score)
            # sentence = torch.tensor(sentence)
            self.score.append(score)
            self.sentence.append(sentence)
        self.score=torch.tensor(self.score)
        self.sentence=torch.tensor(self.sentence)

    def __getitem__(self, item):
        return self.score[item],self.sentence[item]

    def __len__(self):
        return len(self.score)


# test_reader = Reader("./data/SST-cla/dev.txt",0)
# print("max_len",test_reader.max_line)
# print(test_reader.__getitem__(0))
# print(test_reader.__getitem__(1))
# print("reading glove")
# dict={0:"<unk>"}
# with open("./data/SST-cla/glove.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         data = line.strip().split(' ')[0]
#
#         dict[len(dict.keys())]=data
#     s,t=test_reader.__getitem__(0)
#     res=[]
#     for i in t.numpy():
#         res.append(dict[i])
# print(' '.join(res))
# print("res")
