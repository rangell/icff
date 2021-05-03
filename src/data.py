import random
import numpy as np

from IPython import embed


def gen_data(opt):
    
    # data generation hyperparameters
    num_entities = opt.num_entities
    num_mentions = opt.num_mentions
    dim = opt.data_dim
    entity_noise_prob = opt.entity_noise_prob
    mention_sample_prob = opt.mention_sample_prob

    # generate the entities
    entities, mentions, mention_labels = [], [], []
    block_size = dim // num_entities
    for ent_idx in range(num_entities):
        tmp_ent = np.zeros(dim, dtype=int)
        tmp_ent[ent_idx*block_size:(ent_idx+1)*block_size] = 1
        noise_domain = np.where(tmp_ent == 0)[0]
        noise_mask = (np.random.uniform(0, 1, size=noise_domain.shape)
                      < entity_noise_prob)
        noise_idx = noise_domain[noise_mask]
        tmp_ent[noise_idx] = 1
        entities.append(tmp_ent)
    entities = np.vstack(entities)

    # assert entities are non-nested!!!
    assert np.all(np.argmax(entities @ entities.T, axis=1)
                  == np.array(range(entities.shape[0])))

    # generate the mentions
    for _ in range(num_mentions):
        ent_idx = random.randint(0, num_entities-1)
        mention_labels.append(ent_idx)
        while True:
            ent_feat_domain = np.where(entities[ent_idx] == 1)[0]
            ent_feat_mask = (np.random.uniform(0, 1, size=ent_feat_domain.shape)
                             < mention_sample_prob)
            ent_feat_idx = ent_feat_domain[ent_feat_mask]
            sample_mention = np.zeros_like(entities[ent_idx])
            sample_mention[ent_feat_idx] = 1
            subsets = np.all((sample_mention & entities) == sample_mention, axis=1)
            if subsets[ent_idx] and np.sum(subsets) == 1:
                mentions.append(sample_mention)
                break
    mentions = np.vstack(mentions)
    mention_labels = np.asarray(mention_labels)

    # HACK: entity reps are exactly the aggregation of all their mentions
    #        (no more, no less) -> this way we don't need to add attributes
    #        which are not present in the mentions
    for ent_idx in range(num_entities):
        mention_mask = (mention_labels == ent_idx)
        assert np.sum(mention_mask) > 0
        entities[ent_idx] = (np.sum(mentions[mention_mask], axis=0) > 0).astype(int)
        
    return entities, mentions, mention_labels

