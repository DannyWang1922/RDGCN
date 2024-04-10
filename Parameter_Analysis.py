import torch
from RDGCN_Dataset import RDGCNDataset
from model_RDGCN import RDGCNModel, RDGCNEncoder, RDGCNDecoder
from model_RDGCN_v2 import RDGCNModel_v2, RDGCNEncoder_v2, RDGCNDecoder_v2
from model_RDGCN_v3 import RDGCNModel_v3, RDGCNEncoder_v3, RDGCNDecoder_v3

dataset = RDGCNDataset(root="./data")
data = dataset[0]

# model = RDGCNModel(RDGCNEncoder(data=data, in_dims=256, out_dims=64, slope=0.2),
#                       RDGCNDecoder())

model = RDGCNModel_v3(RDGCNEncoder_v3(data=data, in_dims=256, out_dims=64, slope=0.2),
                      RDGCNDecoder_v3())

model.load_state_dict(torch.load("result/result_2/model_state_dict.pth"))
for name, param in model.named_parameters():
    print(f"Layer: {name}")
    print(f"Type: {type(param.data)}")
    print(f"Size: {param.size()}")
    print(f"Values: \n{param.data}\n")