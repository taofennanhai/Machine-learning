import numpy


class Process:

    def __init__(self, input, pre, post):
        self.input = input
        self.pre = pre
        self.post = post

    def process_post(self):

        pass


class VarDeclare():  # 定义变量声明节点
    def __init__(self, var_node, type_node):  # 变量声明由变量和类型组成
        self.var_node = var_node
        self.type_node = type_node


class Type(AST):  # 定义类型节点
    def __init__(self, token):
        self.token = token
        self.name = token.value




class Token:

    pass
