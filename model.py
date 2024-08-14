import torch
import torch.nn.functional as F

from torch.nn import ModuleList, Linear, LSTM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_scatter import scatter


class Net(torch.nn.Module):
    def __init__(self, trunk_list, input_dim, hidden_dim, output_dim,
                 num_layers, dropout, bidirectional, max_level, device):
        super(Net, self).__init__()

        self.trunk_list = trunk_list

        self.bidirectional = bidirectional
        self.dropout = dropout
        self.max_level = max_level
        self.device = device

        self.node_encoder = Linear(input_dim, hidden_dim)

        self.lstms_attr = ModuleList()
        self.lins_attr = ModuleList()

        for level in range(max_level):
            self.lstms_attr.append(
                LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                     dropout=dropout, bidirectional=bidirectional))
            self.lins_attr.append(Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        self.node_encoder.reset_parameters()

        for lstm_a, lin_a in zip(self.lstms_attr, self.lins_attr):
            lstm_a.reset_parameters()
            lin_a.reset_parameters()

    def forward(self, batch_feature):
        batch_size = batch_feature.shape[0]
        num_nodes = batch_feature.shape[1]

        x = batch_feature.reshape(-1, 1)
        x_attr = self.node_encoder(x)

        start_idx = [0]
        h_attr_list = list()

        for level in range(1, self.max_level+1):
            x_attr_list = list()
            trunk_lengths = list()
            num_trunks = list()
            indices = list()

            for i in range(batch_size):
                if level == 1:
                    start_idx.append(start_idx[i] + num_nodes)

                trunks = self.trunk_list[level-1]
                num_trunks.append(len(trunks))
                indices.extend([i] * num_trunks[i])

                for t in trunks:
                    x_attr_list.append(x_attr[start_idx[i]:][torch.tensor(t)])
                    trunk_lengths.append(len(t))

            x_pad = pad_sequence(x_attr_list, batch_first=True, padding_value=0.0)
            x_pack = pack_padded_sequence(x_pad, trunk_lengths, batch_first=True, enforce_sorted=False)
            output, (h_n, c_n) = self.lstms_attr[level-1](x_pack)

            h_last = h_n[-1, :, :] + h_n[-2, :, :] if self.bidirectional else h_n[-1, :, :]
            h_trunk = scatter(h_last, torch.tensor(indices).to(self.device),
                              dim=0, dim_size=batch_size, reduce='sum')
            h_attr = F.dropout(h_trunk, self.dropout, training=self.training)
            h_attr_list.append(h_attr)

        score_over_layer = 0
        # combine all trunk representations at various levels to create a representation of the brain tree
        for level, h_attr in enumerate(h_attr_list):
            score_over_layer += F.dropout(self.lins_attr[level](h_attr), self.dropout, training=self.training)

        return score_over_layer
