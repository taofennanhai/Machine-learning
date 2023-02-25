# import collections
# import math
# import torch
# from torch import nn


# input_dim = 1
# output_dim = 1
#
#
# test = torch.zeros(20, 198, 1)
#
#
# class dataset(nn.Module):
#     def __init__(self):
#         pass
#
#
# decoder = nn.GRU(input_size=input_dim, hidden_size=output_dim, batch_first=True, num_layers=2)
#
# # h: shape = [num_layers * num_directions, batch, hidden_size]的张量 h包含的是句子的最后一个单词的隐藏状态
# # c: 与h的形状相同，它包含的是在当前这个batch_size中的每个句子的初始细胞状态。h,c如果不提供，那么默认是0
# test, h = decoder(test)

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

data = {'result': 'prediction":"0","proba":"0.9975490196078431","log_line":"187.167.57.27 - - [15/Dec/2018:03:48:45 -0800] \"GET /honeypot/Honeypot%20-%20Howto.pdf HTTP/1.1\" 200 1279418 \"http://www.secrepo.com/\" \"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.24 (KHTML, like Gecko) Chrome/61.0.3163.128 Safari/534.24 XiaoMi/MiuiBrowser/9.6.0-Beta\"'}
host = ('localhost', 8000)


class My_Server(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)

        # 发给请求客户端的响应数据
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        self.send_response(200)

        datas = self.rfile.read(int(self.headers['content-length']))
        print('headers', self.headers)
        print("-->> post:", self.path, self.client_address)
        print(datas)

        # 发给请求客户端的响应数据
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


if __name__ == '__main__':
    server = HTTPServer(host, My_Server)
    print("server启动@ : %s:%s" % host)

    server.serve_forever()


if __name__ == "__main__":
    main()






