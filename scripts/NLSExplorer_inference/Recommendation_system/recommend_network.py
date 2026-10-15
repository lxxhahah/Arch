import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba, MambaConfig

config = MambaConfig(d_model=16, n_layers=2)
model = Mamba(config)


class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        # The linear layer that maps from hidden state space to tag space
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        # self.ll = nn.Embedding(500, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, embding):
        rate_matrix = self.atten_Matrix(embding)
        rate_matrix = self.relu(rate_matrix)
        att_rate = F.softmax(rate_matrix,dim=1)
        len = rate_matrix.size()[1]
        result = (embding * att_rate).sum(1) / math.sqrt(len)
        result = self.layer_norm(result)
        return result, att_rate

class gnn_classifier(nn.Module):
    def __init__(self):
        super(gnn_classifier, self).__init__()
        hidden_dim = 1280
        lstm_dim = int(hidden_dim / 4)
        cls_dim = lstm_dim * 2
        conv_dim = 256
        self.cls = nn.Linear(cls_dim, 1)
        self.lstm = nn.LSTM(int(hidden_dim), lstm_dim, 1, bidirectional=True, batch_first=True)
        self.sigmoi = nn.Sigmoid()
        self.att_fu = Attention(cls_dim)
        config = MambaConfig(d_model=int(hidden_dim / 2), n_layers=12)
        self.mamb = Mamba(config)


        config_1 = MambaConfig(d_model=hidden_dim, n_layers=6) # for enhanced esmbedding
        self.mamb1 = Mamba(config_1)


   
    def forward(self, embedding, segments):
        """
            embedding is a tensor -> 
            segments
        """
        batch_size, max_len, _ = embedding.size()
        embedding = embedding.cuda()

        results = []
        emb_li = []

        embedding = self.mamb1(embedding)+embedding # additionally Mamba
        # 逐样本计算
        for batch_idx in range(batch_size): # actually batch_size = 1
            emb_batch = embedding[batch_idx].unsqueeze(0)  # (1, max_len, 1280)
            segment_batch = segments[batch_idx]  # respond 
            
            batch_results = []

            for segment in segment_batch:
                start, end = int(segment[0].item()), int(segment[1].item())
                if start == 0 and end == 0:
                    continue

                if end + 1 <= max_len:
                    posi_emb = emb_batch[:, start:end + 1, :]
                else:
                    posi_emb = emb_batch[:, start:max_len, :]

                # posi_emb: (1, seg_len, 1280) 
                lstm_out, _ = self.lstm(posi_emb)
                lstm_out = self.mamb(lstm_out) + lstm_out
                att_out, _ = self.att_fu(lstm_out)
                emb_li.append(att_out)
                cls_out = self.sigmoi(self.cls(F.relu(att_out)))
                # cls_out: [1, 1] 
                batch_results.append(cls_out)

            if len(batch_results) > 0:
                batch_results = torch.cat(batch_results, dim = 1)  # (1, num_segments)
                results.append(batch_results)

        if len(results) > 0:
            max_len_result = max([res.size(1) for res in results])
            results = [F.pad(res, (0, max_len_result - res.size(1)), "constant", 0) for res in results]
            results = torch.cat(results, dim=0)
        else:
            results = torch.tensor([]).cuda()
        
        return results, emb_li
    

