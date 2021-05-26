import os
import sys
import json
import pickle
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from IPython import embed


# Get the file we would like to process
try:
    assert len(sys.argv) == 3
except:
    print('Usage: python process-canopy.py [CANOPY_DIR] [OUTFILE]')
    print('Example: python process-canopy.py canopy/robinson_h rexa-robinson_h.pkl')

canopy_dir = sys.argv[1]
outfile = sys.argv[2]

assert os.path.isdir(canopy_dir)

gt_ent2idx = {}
feat2idx = {}

def get_or_create_idx(dictionary, key):
    idx = dictionary.get(key, None)
    if idx is None:
        idx = len(dictionary)
        dictionary[key] = idx
    return idx

# Read in mentions
ment_feat_idxs = []
labels = []
with open(f'{canopy_dir}/ments.json', 'r') as f:
    for line in f:
        ment = json.loads(line)
        feats = ment['pack']
        lbl = ment['gt']
        feat_idxs = []
        for f in feats:
            if f[:2] == 't:':
                title_feats = f[2:].lower().split(' ')
                for tfs in title_feats:
                    feat_str = 't:' + tfs
                    feat_idxs.append(get_or_create_idx(feat2idx, feat_str))
            else:
                feat_idxs.append(get_or_create_idx(feat2idx, f))
        ment_feat_idxs.append(feat_idxs)
        labels.append(get_or_create_idx(gt_ent2idx, lbl))

# Convert everything to csr
def lol2csr(lol, shape):
    row = []
    col = []
    for r, col_idxs in enumerate(lol):
        for c in col_idxs:
            row.append(r)
            col.append(c)
    data = [1] * len(row)
    return csr_matrix((data, (row, col)), shape=shape)

mentions = lol2csr(ment_feat_idxs, (len(ment_feat_idxs), len(feat2idx)))
mention_labels = np.array(labels)

# build the gold_entities
entities = []
for lbl in np.unique(mention_labels):
    ent_mask = (mention_labels == lbl)
    ent_col = list(set(mentions[ent_mask].tocoo().col.tolist()))
    ent_row = [0] * len(ent_col)
    ent_data = [1] * len(ent_col)
    entity = csr_matrix(
        (ent_data, (ent_row, ent_col)), shape=(1, mentions.shape[1])
    )
    entities.append(entity)

gold_entities = sp.vstack(entities)

print('Num mentions: {}'.format(mentions.shape[0]))
print('Num entities: {}'.format(gold_entities.shape[0]))
print('Num features: {}'.format(gold_entities.shape[1]))

assert gold_entities.shape[1] == mentions.shape[1]

# dump the data
with open(outfile, 'wb') as f:
    pickle.dump((gold_entities, mentions, mention_labels), f)
