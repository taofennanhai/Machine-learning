import re
import copy
class Var():
    def __init__(self,name,type,val = None):
        self.name = name
        self.type = type
        self.val = val
    # def getVal(self, name, varMap): # 使用varMap执行输入
    #   pass        
    def __repr__(self):
        return f'Var({self.name},{self.type})'
class DataStruct():
    def __init__(self,txt):
        self.txt = txt
        self.vars = [] 
        割 = txt.split(',')
        割.reverse()
        当前type = None
        for 某 in 割:            
            此 = 某.split(':')
            if len(此)>1:
                当前type = 此[1].strip()
            self.vars.append(Var(此[0].strip(),当前type))
        self.vars.reverse()
    def param列表(self):
        return [v.type for v in self.vars]
    def __repr__(self):
        return f'''DataStruct({self.vars}'''

class Function():
    funcs = []
    ops = []
    def __init__(self, name, func):
        self.func = func
        self.name = name
        Function.funcs.append(self)
    def exec(self,vars):
        return self.func(vars)
    @classmethod
    def expr(cls):
        names = [s.name for s in cls.funcs]
        return fr'({"|".join(names)})\s*\(([^\(\)]*?)\)'
    def 调用(cls, name, varList):
        for f in cls.ops: # 运算符
            if f.name == name:
                return f.func(varList)
        for f in cls.func: # 函数
            if f.name == name:
                return f.func(varList)
        # return None
    

class Expr():
    functions = Function.funcs
    exps = {
    'or': fr'(.*?)(?!<=\w)(or)(?!\w)(.*)' , 
    'and' : fr'(.*?)(?!<=\w)(and)(?!\w)(.*)' , 
    'Rel-op':fr'(.*?)(<>|<=|>=|=|<|>)(.*)' , 
    'Bin-op':fr'(.*?)(\*|\/|\+|\-)(.*)'
    }
    def __init__(self, txt:str):
        self.txt = txt.strip()
        self.subExpr = []
        self.op =  ''
        self.data = None
        l = re.findall(Expr.exps['or'],txt)
        if self.data is None and len(l)>0:
            self.data = l[0]
        l = re.findall(Expr.exps['and'],txt)
        if self.data is None and len(l)>0:
            self.data = l[0]
        l = re.findall(Expr.exps['Rel-op'],txt)
        if self.data is None and len(l)>0:
            self.data = l[0]
        l = re.findall(Expr.exps['Bin-op'],txt)
        if self.data is None and len(l)>0:
            self.data = l[0]
        # l = re.findall(Expr.exps['Bin-op'],txt)
        # if self.SecondQuestionData is None and len(l)>0:
        #     self.SecondQuestionData = l[0]
        if self.data == None:
            self.data = txt.strip()
            self.op = 'val'
        else:
            # self.SecondQuestionData = self.SecondQuestionData.strip()
            self.op = self.data[1]
            self.subExpr.append(Expr(self.data[0]))
            self.subExpr.append(Expr(self.data[2]))
        # print('===')
        # print(self.txt)
        # print(self.op)
        # print(self.SecondQuestionData)
        # print()

    def 测(self,varMap, resMap):
        if self.op == 'or':
            if self.subExpr[0].judge(varMap,resMap):
                self.subExpr[0].result(varMap,resMap)
            else:
                self.subExpr[1].result(varMap,resMap)
            # print('res',r)
        print(resMap)

    def result(self,varMap, resMap): # 結果はなんなの？         
        # print('judge', self.txt, self.op,self.subExpr)
        # print("resMap!!!!!",resMap)
        res = None
        if len(self.subExpr) == 0:
            if self.data.startswith('true('):
                return varMap.get(self.data[5:-1]) is not None            
            try:
                res = int(self.data)
            except:
                res = varMap.get(self.data)
                if res is None:
                    res = None
                else:
                    res = int(res)
            # print('000',res)
            return res
        param1,param2 = None,None
        if self.op == 'or':
            if self.subExpr[0].judge(varMap,resMap):
                param1 = self.subExpr[0].result(varMap,resMap)
            else:
                param2 = self.subExpr[1].result(varMap,resMap)
        else:
            param1 = self.subExpr[0].result(varMap,resMap)
            param2 = self.subExpr[1].result(varMap,resMap)
        # print('params',param1,param2,self.op)
        if resMap is not None and self.op == '=':
            if param1 is None and param2 is not None:
                resMap[self.subExpr[0].data] = param2
        elif self.op == 'or':
            res = param1 or param2
        elif self.op == 'and':
            res = param1 and param2
        elif param1 is None or param2 is None:
            res = None
        elif self.op == '=':
            res = param1 == param2

        elif self.op == '<>':            
            res = param1 != param2
        elif self.op == '<=':
            res = param1 <= param2
        elif self.op == '>=':
            res = param1 >= param2
        elif self.op == '<':
            res = param1 < param2
        elif self.op == '>':
            res = param1 > param2
        # el
        elif self.op == '+':
            res = param1 + param2
        elif self.op == '-':
            res = param1 - param2
        elif self.op == '*':
            res = param1 * param2
        elif self.op == '/':
            res = param1 / param2
        # print('judge', self.txt, self.op,self.subExpr)
        # print(res)
        return res
    def judge(self,varMap,resMap=None): # 出来るのか？ True , False , None
        # print('judge', self.txt, self.op,self.subExpr)
        # print("resMap!!!!!",resMap)
        res = None
        if len(self.subExpr) == 0:
            if self.data.startswith('true('):
                return varMap.get(self.data[5:-1]) is not None            
            try:
                res = int(self.data)
            except:
                res = varMap.get(self.data)
                if res is None:
                    res = None
                else:
                    res = int(res)
            # print('000',res)
            return res

        param1 = self.subExpr[0].judge(varMap,resMap)
        param2 = self.subExpr[1].judge(varMap,resMap)
        # print('params',param1,param2,self.op)
        if resMap is not None and self.op == '=':
            if param1 is None and param2 is not None:
                res = True
        elif self.op == 'or':
            res = param1 or param2
        elif self.op == 'and':
            res = param1 and param2
        elif param1 is None or param2 is None:
            res = None
        elif self.op == '=':
            res = param1 == param2

        elif self.op == '<>':            
            res = param1 != param2
        elif self.op == '<=':
            res = param1 <= param2
        elif self.op == '>=':
            res = param1 >= param2
        elif self.op == '<':
            res = param1 < param2
        elif self.op == '>':
            res = param1 > param2
        # el
        elif self.op == '+':
            res = param1 + param2
        elif self.op == '-':
            res = param1 - param2
        elif self.op == '*':
            res = param1 * param2
        elif self.op == '/':
            res = param1 / param2
        # print('judge', self.txt, self.op,self.subExpr)
        # print(res)
        return res
    def __repr__(self):
        return self.txt


class Process():
    """くそやろ"""
    
    def __init__(self, txt):
        # 四つの部分
        # print(txt)
        列表 = re.findall('(?<!\\w)process(?!\\w)\\s(\\w+)\\((.*)\\)(.*)\\n\\s*pre((?:(?:.|\\s)(?!post))*)\\s*post((?:(?:.|\\s)(?!end_process))*)(?:(?:.|\\n)*)(?<!\\w)end_process(?!\\w)', txt)
        # print(列表)
        self.name,self.input,self.output,self.pre,self.post = 列表[0]    
        self.name = self.name.strip()
        self.input= self.input.strip()
        self.output= self.output.strip()
        self.pre = self.pre.strip()
        self.post = self.post.strip()
        # ここから、基本的の四つの部分が終わりました。
        # 先ずは、アウトプットのプロセス
        self.outStruct = [DataStruct(i) for i in self.output.split('|')]
        self.inStruct = [DataStruct(i) for i in self.input.split('|')]
        # self.outStruct = DataStruct(self.output)

        varMap = {}
    def 运行(self, varMap):        
        # プレコンディショニングの解析
        pre = Expr(self.pre)
        if not pre.judge(varMap):
            raise "不通过"
        # インプットの解析
        # 何このバカバカしいロジスティクス、ま、インプットはマップにするでいいの
        resMap = {}
        post = Expr(self.post)
        post.result(varMap,resMap)
        # post.测(varMap, resMap)
        # print('resMap',resMap)
        return resMap

    def input列表(self):
        return [s.param列表() for s in self.inStruct]

    def __repr__(self):
        return f'''Process(
    name\t:\t{self.name}
    input\t:\t{self.input}
    inputStruct:
{self.inStruct}
    output\t:\t{self.output}
    outputStruct:
{self.outStruct}
    pre\t\t:\t{self.pre}
    post\t:\t{self.post}
)'''


txt = [   '''process A(x: int, y: int) Z: int, w: int
prex> 0 and y> 0
post z=x+ y and w=x-y
end_process''',
    '''process A(x, y: int) z, w: int 
pre  x>0   and  y>0
post z =x+ y and w=x- y
end_process''',
    '''process B(x: int | y: int) z: int
pre x> 0 or y > 0
post z=x+ 1 or z=y- 1
end_process''',
    '''process B(x: int | y: int) z: intparams 
pre x> 0 or true(y)
post z=x+1 or z=y-1
end_process''',
    '''process C(x: int) z: int| w: int
pre x>0
post x < 10 and z=x+ 1 or x>= 10 and w=x*2
end_process''',
]

print('=====')
varMap = {
    'x' : 9 , 
    'y' : 100
}
print(varMap)
p = Process(txt[0])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')
    
print('=====')
varMap = {
    # 'x' : 9 , 
    'y' : 100
}
print(varMap)
p = Process(txt[0])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : -9 , 
    'y' : 100
}
print(varMap)
p = Process(txt[0])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : 9 , 
    'y' : 100
}
print(varMap)
p = Process(txt[1])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    # 'x' : 9 , 
    'y' : 100
}
print(varMap)
p = Process(txt[1])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : -9 , 
    'y' : 100
}
print(varMap)
p = Process(txt[1])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

###################################
print('=====')
varMap = {
    'x' : -9 , 
    # 'y' : 100
}
print(varMap)
p = Process(txt[2])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : 9 , 
    # 'y' : 100
}
print(varMap)
p = Process(txt[2])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    # 'x' : -9 , 
    'y' : 100
}
print(varMap)
p = Process(txt[2])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    # 'x' : -9 , 
    'y' : 0
}
print(varMap)
p = Process(txt[2])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')
#####################################
print('=====')
varMap = {
    # 'x' : -9 , 
    'y' : 0
}
print(varMap)
p = Process(txt[3])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : -9 , 
    # 'y' : 0
}
print(varMap)
p = Process(txt[3])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : 9 , 
    # 'y' : 0
}
print(varMap)
p = Process(txt[3])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')
##################
print('=====')
varMap = {
    'x' : 9 , 
    # 'y' : 0
}
print(varMap)
p = Process(txt[4])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : -1 , 
    # 'y' : 0
}
print(varMap)
p = Process(txt[4])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')

print('=====')
varMap = {
    'x' : 20 , 
    # 'y' : 0
}
print(varMap)
p = Process(txt[4])
print(p)
try:
    print(p.运行(varMap))
except:
    print('不通过')