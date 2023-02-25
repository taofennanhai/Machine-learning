import argparse


parse = argparse.ArgumentParser()    # 创建一个实例
parse.add_argument("--dev_set")
parse.add_argument("--train_set")    # 增加一个参数，起名train_set,加--，不按顺序解析，在命令中不必有
parse.add_argument("test_set")    # 增加一个参数，起名test_set,不加-，按顺序解析，在命令中必须有
args = parse.parse_args()    # 处理我们刚刚增加的参数，方便后面使用

print("train_set ", args.train_set)    # 打印我们刚增加的参数
print("test_set ", args.test_set)    # 打印我们刚增加的参数
print("dev_set ", args.dev_set)
