import torch
from RDGCN_Dataset import RDGCNDataset
from models.model_RDGCN import RDGCNModel, RDGCNEncoder, RDGCNDecoder
from models.model_RDGCN_v2 import RDGCNModel_v2, RDGCNEncoder_v2, RDGCNDecoder_v2
from models.model_RDGCN_v3 import RDGCNModel_v3, RDGCNEncoder_v3, RDGCNDecoder_v3
from models.model_RDGCN_v4 import RDGCNModel_v4, RDGCNEncoder_v4, RDGCNDecoder_v4
from models.model_RDGCN_v5 import RDGCNModel_v5, RDGCNEncoder_v5, RDGCNDecoder_v5

dataset = RDGCNDataset(root="./data")
data = dataset[0]

# model = RDGCNModel(RDGCNEncoder(data=data, in_dims=256, out_dims=64, slope=0.2),
#                       RDGCNDecoder())

model = RDGCNModel_v5(RDGCNEncoder_v5(data=data, in_dims=256, out_dims=64, slope=0.2, dropout=0.5, aggr="mean"),
                      RDGCNDecoder_v5())

model.load_state_dict(torch.load("result/best_model/model_state_dict.pth", map_location=torch.device('cpu')))
for name, param in model.named_parameters():
    print(f"Layer: {name}")
    print(f"Type: {type(param.data)}")
    print(f"Size: {param.size()}")
    print(f"Values: \n{param.data}\n")