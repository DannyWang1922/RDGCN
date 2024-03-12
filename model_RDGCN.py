import torch


class RDGCN(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, diseases, mirnas):
        h = self.encoder(G)
        h_diseases = h[diseases]
        h_mirnas = h[mirnas]
        out2 = self.decoder(h_diseases, h_mirnas)
        return out2


class RDGCN_Encoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, diseases, mirnas):
        h = self.encoder(G)
        h_diseases = h[diseases]
        h_mirnas = h[mirnas]
        out2 = self.decoder(h_diseases, h_mirnas)
        return out2
