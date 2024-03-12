import warnings

import numpy as np

from main import RDGCN

if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  auc, acc, pre, recall, f1, fprs, tprs, aupr = RDGCN(directory='data',
                                                      epochs_num=100,
                                                      aggregator='GraphSAGE',  # 'GraphSAGE'
                                                      embedding_size=256,
                                                      layers=2,
                                                      dropout=0.7,
                                                      slope=0.2,  # LeakyReLU
                                                      lr=0.001,
                                                      wd=2e-3,
                                                      random_seed=126)

  print('seed: %.4f \n' % (126),
        '-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc), np.std(auc)),
        'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc), np.std(acc)),
        'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre), np.std(pre)),
        'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall), np.std(recall)),
        'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1), np.std(f1)))