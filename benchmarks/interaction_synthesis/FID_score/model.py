from argparse import ArgumentParser
import torch
import torch.nn as nn
from layers import ResnetPointnet, PositionalEncoding, TransformerEncoder
from sublayers import TransformerEncoderLayerQaN

class ActionRecogNet(nn.Module):
    def __init__(self, args):
        super(ActionRecogNet, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        self.bodyEmbedding1 = nn.Linear(24*3, num_channels)
        self.bodyEmbedding2 = nn.Linear(24*3, num_channels)
        self.objEmbedding = ResnetPointnet(num_channels, 3, num_channels)
        self.PositionalEmbedding = PositionalEncoding(d_model=num_channels, dropout=args.dropout)
        
        from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
        seqTransEncoderLayer1 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        seqTransEncoderLayer2 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        seqTransEncoderLayer3 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        seqTransEncoderLayer4 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        seqTransEncoderLayer5 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        seqTransEncoderLayer6 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        seqTransEncoderLayer7 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        seqTransEncoderLayer8 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=True)
        # seqTransEncoderLayer = nn.ModuleList([seqTransEncoderLayer1, seqTransEncoderLayer2, seqTransEncoderLayer3, seqTransEncoderLayer4, seqTransEncoderLayer5, seqTransEncoderLayer6, seqTransEncoderLayer7, seqTransEncoderLayer8])
        seqTransEncoderLayer = nn.ModuleList([seqTransEncoderLayer1])
        self.encoder = TransformerEncoder(seqTransEncoderLayer)

        self.finalLinear = nn.Linear(num_channels, 9)
        
        self.linear = nn.Linear(num_channels, num_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        """
        x: shape = (B, T, 24*3+24*3+N_point*3)
        """
        
        B, T, _ = x.shape
        
        body1 = self.bodyEmbedding1(x[:, :, :24*3])  # (B, T, num_channels)
        body1 = self.act(body1)
        body2 = self.bodyEmbedding2(x[:, :, 24*3:24*3+24*3])  # (B, T, num_channels)
        body2 = self.act(body2)
        obj = self.objEmbedding(x[:, :, 24*3+24*3:].reshape(B*T, -1, 3)).reshape(B, T, -1)  # (B, T, num_channels)
        obj = self.act(obj)
        # print(body1.shape, body2.shape, obj.shape)
        
        embedding = body1 + body2 + obj  # (B, T, num_channels)
        # embedding = self.PositionalEmbedding(embedding)  # (B, T, num_channels)
        # embedding = self.encoder(embedding)  # (B, T, num_channels)
        
        e1 = self.linear(embedding)
        e2 = self.act(e1)
        e3 = self.linear(e2)
        e4 = self.act(e3)

        f_seq = e4.max(dim=1)[0]  # (B, num_channels)
        pred0 = self.finalLinear(f_seq)  # (B, N_action)
        pred = self.act(pred0)
        
        return pred, f_seq
    

if __name__ == "__main__":
    
    # args
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--latent_usage", type=str, default='memory')
    parser.add_argument("--template_type", type=str, default='zero')
    parser.add_argument('--star_graph', default=False, action='store_true')
    args = parser.parse_args()
    
    model = ActionRecogNet(args)
    model.to("cuda:0")
    x = torch.randn((32, 120, 24*3+24*3+32*3)).to("cuda:0")
    y = model(x)
    print(y.shape)
