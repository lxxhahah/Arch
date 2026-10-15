import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()

        self.ll =    nn.Embedding(500,hidden_dim) # may not use
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, embding):
        """
            embding: [bz, seq_len, 1280]
            return:
                    sum_:  [bz, 1280] (整体表征)
                    att_rate: 注意力分布图
        """
        rate_matrix = self.atten_Matrix(embding) # [bz, seq_len, 1]
        rate_matrix = self.relu(rate_matrix)
        # norm-----
        att_rate = F.softmax(rate_matrix,dim=1) # [bz, seq_len, 1]
        # 加权求和并缩放
        lll= rate_matrix.size()[1] 
        sum_ = (embding*att_rate).sum(1)/math.sqrt(lll) # [bz, 1280]
        # len_emb = self.ll(torch.tensor(lenttth).cuda()).unsqueeze(0)
        # sum_ = len_emb*sum_
        # print(sum.size())
        sum_ = self.layer_norm(sum_)
        return sum_,att_rate

class LSTMTagger(nn.Module):
    """
        P: 输出概率预测 
        attention_dis: store each layer summary attn_dist
        total_attention_store: 每个头的: each element is also a list---- a single layer coresponding all attention_distribution
        final_att: 最终综合attention
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.Att_config = [64]*16
        self.dropout = nn.Dropout(p=0.1)
        self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,2,bidirectional=True,batch_first = True)
        
        # The linear layer that maps from hidden state space to tag space
        real_dim = hidden_dim


        Att_li = []
        for fig in self.Att_config:
            sub_li = []
            for k in range(fig):
                sub_li.append(Attention(real_dim))
            Att_li.append(nn.ModuleList(sub_li))
        self.Att_li = nn.ModuleList(Att_li)    
        self.AAt = Attention(real_dim)
        
        # 压缩得到的vec维度
        pro_li = []
        for fig in self.Att_config:
            sub_li = []
            to_dim = real_dim/fig
#             print(to_dim)
            for k in range(fig):
                sub_li.append(nn.Linear(real_dim,int(to_dim)))
            pro_li.append(nn.ModuleList(sub_li))
        self.pro_li = nn.ModuleList(pro_li)    
        
        # 用于将综合好的embedding投影到高维以方便产生乘积
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(1,real_dim))
        self.project_li = nn.ModuleList(project_li)
                
        #线性变换填充维度
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(real_dim,real_dim))
        self.FI_li = nn.ModuleList(project_li)

        #压缩一下避免爆炸
        set_dim = int(real_dim/4)
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(real_dim,set_dim))
        self.Set_li = nn.ModuleList(project_li)
       
        #最后投影到概率上面去
        length = (len(self.Att_config))*set_dim+real_dim
        self.hidden2p = nn.Linear(length,1)

        #norm 层
        norm_li = []
        for i in range(len(self.Att_config)):
            norm_li.append(nn.LayerNorm(real_dim))
        self.norm_li = nn.ModuleList(norm_li)
        
    def forward(self, embding):
        batch_size = (embding.size()[0])
        vec_store = [] # store each layer summary feature
        t_emb = embding
        #储存一开始的emb_
        origin_emb = t_emb
        attention_dis = [] # store each layer summary attn_dist

        #需要把每个单元的注意力模式都提取出来
        #先进行提取，其余的我们之后再进行处理
        total_attention_store = [] # each element is also a list---- a single layer attention_distribution

        for i,fig in enumerate(self.Att_config): # process single layer(64 BAU)
            vec_s = [] # store output feature
            att_s = [] # store att_rate
            for k in range(fig):
                vec,att_ = self.Att_li[i][k](t_emb)
                vec = self.pro_li[i][k](vec)
                vec = F.relu(vec)
                vec_s.append(vec)
                att_s.append(att_)
            total_attention_store.append(att_s)
            # compute summary attn for current layer
            att_s = torch.stack(att_s).squeeze(3)
            sum_att = att_s.sum(0).unsqueeze(2)
            # store each layer summary attn_dist
            attention_dis.append(att_s.sum(0).unsqueeze(2))
            # 生成下一层的embedding
            sum_att = self.project_li[i](sum_att)
            t_emb = (t_emb*sum_att)+t_emb
            t_emb = self.norm_li[i](t_emb)

            #压缩当前层特征
            vec_s =  torch.stack(vec_s)
            vec_s = vec_s.transpose(0,1)

            #下面是生成每个层的total_embedding方法
            z = vec_s
            z = z.permute(0,2,1)
            batch = z.size()[0]
            output = z.reshape(batch,-1)
            #线性变换后填充维度
            output = F.relu(self.FI_li[i](output))+output
            output = F.relu(self.Set_li[i](output))
            vec_store.append(output)
            
        # 拼接所有层的压缩特征
        ott = torch.stack(vec_store).transpose(0,1).reshape(batch_size,-1)
        # 计算最终注意力与聚合特征
        sum_, final_att = self.AAt(t_emb)
        # 合并所有特征
        um_ = torch.cat((sum_,ott),1)       
        um_ = self.dropout(um_)  
        # 输出概率预测  
        P = torch.sigmoid(self.hidden2p(um_)) # [batch, 1]
      
        return P, attention_dis, total_attention_store, final_att   
