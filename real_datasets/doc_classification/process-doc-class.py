import os
import sys
import pickle
from functools import reduce
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sparse_dot_mkl import dot_product_mkl

from IPython import embed


# Get the file we would like to process
try:
    assert len(sys.argv) == 3
except:
    print('Usage: python process-doc-class.py [INFILE] [OUTFILE]')
    print('Example: python process-doc-class.py datasets/20ng-test-stemmed.txt 20-test-stemmed.dataset.pkl')

infile = sys.argv[1]
outfile = sys.argv[2]

assert os.path.isfile(infile)

# read the file in
labels, examples = [], []
with open(infile, 'r') as f:
    for line in f:
        lbl, feats = line.split('\t')
        if feats == '\n':
            continue
        feats = feats.strip().split(' ')
        labels.append(lbl)
        examples.append(" ".join(feats))

# learn the features
#vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=0.001)
vectorizer = CountVectorizer(ngram_range=(1,3), min_df=0.001)
mentions = vectorizer.fit_transform(examples)
b_mentions = mentions.astype(bool).astype(int)

# get mention labels
label2idx = {label : i for i, label in enumerate(set(labels))}
mention_labels = np.array([label2idx[lbl] for lbl in labels])

# build the gold_entities
entities = []
for lbl in np.unique(mention_labels):
    ent_mask = (mention_labels == lbl)
    ent_col = list(set(b_mentions[ent_mask].tocoo().col.tolist()))
    ent_row = [0] * len(ent_col)
    ent_data = [1] * len(ent_col)
    entity = csr_matrix(
        (ent_data, (ent_row, ent_col)), shape=(1, mentions.shape[1])
    )
    entities.append(entity)

gold_entities = sp.vstack(entities)

## check non-nested and distinguishability data assumptions
#_np_gold_entities = gold_entities.toarray()
#assert np.all(np.argmax(_np_gold_entities @ _np_gold_entities.T, axis=1)
#              == np.array(range(_np_gold_entities.shape[0])))
#_np_gold_entities = _np_gold_entities.astype(bool)
#
#filtered_b_mentions = []
#filtered_mentions = []
#filtered_mention_labels = []
#
#for b_mention, mention, mention_label in zip(b_mentions, mentions, mention_labels):
#    _np_mention = b_mention.toarray().reshape(-1,).astype(bool)
#    subsets = np.all((_np_mention & _np_gold_entities) == _np_mention, axis=1)
#    #assert np.sum(subsets) == 1
#    if np.sum(subsets) == 1:
#        filtered_b_mentions.append(b_mention)
#        filtered_mentions.append(mention)
#        filtered_mention_labels.append(mention_label)
#
#assert len(filtered_mentions) == len(filtered_mention_labels)
#
#print('Unfiltered # mentions: {}'.format(mentions.shape[0]))
#print('Filtered # mentions: {}'.format(len(filtered_mentions)))
#
#b_mentions = sp.vstack(filtered_b_mentions)
#mentions = sp.vstack(filtered_mentions)
#mention_labels = np.array(filtered_mention_labels)
#
## build the gold_entities (again!)
#entities = []
#
#assert np.array_equal(np.unique(mention_labels), np.arange(gold_entities.shape[0]))
#
#for lbl in np.unique(mention_labels):
#    ent_mask = (mention_labels == lbl)
#    ent_col = list(set(b_mentions[ent_mask].tocoo().col.tolist()))
#    ent_row = [0] * len(ent_col)
#    ent_data = [1] * len(ent_col)
#    entity = csr_matrix(
#        (ent_data, (ent_row, ent_col)), shape=(1, mentions.shape[1])
#    )
#    entities.append(entity)
#
#gold_entities = sp.vstack(entities)
#
#
##### filter mentions by separability
#LB_DIFF = 30
#feat_overlap = dot_product_mkl(
#    b_mentions.astype(float), gold_entities.T.astype(float), dense=True
#)
#top2_overlap_diffs = np.diff(np.sort(feat_overlap)[:,-2:])
#mention_keep_mask = (top2_overlap_diffs > LB_DIFF).reshape(-1,)
#
## take the top classes
#uniq_labels, label_counts = np.unique(mention_labels[mention_keep_mask], return_counts=True)
##keep_mention_labels = uniq_labels[np.argsort(-label_counts)[:2]]
#keep_mention_labels = uniq_labels[label_counts > 3]
#mention_keep_mask = mention_keep_mask & np.isin(mention_labels, keep_mention_labels)
#
#entities = []
#lbl_remap = {}
#for new_lbl, lbl in enumerate(keep_mention_labels):
#    lbl_remap[lbl] = new_lbl
#    ent_mask = (mention_labels == lbl) & mention_keep_mask
#    ent_col = list(set(b_mentions[ent_mask].tocoo().col.tolist()))
#    ent_row = [0] * len(ent_col)
#    ent_data = [1] * len(ent_col)
#    entity = csr_matrix(
#        (ent_data, (ent_row, ent_col)), shape=(1, mentions.shape[1])
#    )
#    entities.append(entity)
#
#gold_entities = sp.vstack(entities)
#b_mentions = mentions[mention_keep_mask]
#mentions = mentions[mention_keep_mask]
#mention_labels = mention_labels[mention_keep_mask]
#
#v_lbl_remap = np.vectorize(lambda x : lbl_remap[x])
#mention_labels = v_lbl_remap(mention_labels)
#
#print('Further filtered # mentions: {}'.format(mentions.shape[0]))
#print('Num entities: {}'.format(gold_entities.shape[0]))
#
## check to make sure all assumptions are still satisfied
#_np_gold_entities = gold_entities.toarray()
#assert np.all(np.argmax(_np_gold_entities @ _np_gold_entities.T, axis=1)
#              == np.array(range(_np_gold_entities.shape[0])))
#_np_gold_entities = _np_gold_entities.astype(bool)
#
#for b_mention, mention, mention_label in zip(b_mentions, mentions, mention_labels):
#    _np_mention = b_mention.toarray().reshape(-1,).astype(bool)
#    subsets = np.all((_np_mention & _np_gold_entities) == _np_mention, axis=1)
#    assert np.sum(subsets) == 1


# dump the data
with open(outfile, 'wb') as f:
    pickle.dump((gold_entities, mentions, mention_labels), f)
