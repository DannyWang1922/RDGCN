from main import RDGCN
import argparse


# RDGCN(epochs_num=100,
#       n_splits=5,
#       batch_size=256,
#       lr=0.001,
#       weight_decay=0.005,
#       in_dims=256,
#       out_dims=64,
#       slope=0.2,
#       dropout=0.7,
#       random_seed=42)

def main():
    parser = argparse.ArgumentParser(description='RDGCN Configuration')
    parser.add_argument('--epochs_num', type=int, default=100, help='Number of epochs')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for cross-validation')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--in_dims', type=int, default=256, help='Input dimensions')
    parser.add_argument('--out_dims', type=int, default=64, help='Output dimensions')
    parser.add_argument('--slope', type=float, default=0.2, help='Slope for leaky ReLU')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--disjoint_train_ratio', type=float, default=0.5, help='disjoint_train_ratio')
    parser.add_argument('--aggr', type=str, default='mean', help='feature aggregation method')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    RDGCN(epochs_num=args.epochs_num,
          n_splits=args.n_splits,
          batch_size=args.batch_size,
          lr=args.lr,
          weight_decay=args.weight_decay,
          in_dims=args.in_dims,
          out_dims=args.out_dims,
          slope=args.slope,
          dropout=args.dropout,
          disjoint_train_ratio=args.disjoint_train_ratio,
          aggr=args.aggr,
          random_seed=args.random_seed)


if __name__ == '__main__':
    main()
