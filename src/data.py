import random
import numpy as np


def gen_data(num_entities, num_mentions, dim):
    entities, mentions, mention_labels = [], [], []
    block_size = dim // num_entities
    for ent_idx in range(num_entities):
        tmp_ent = np.zeros(dim, dtype=int)
        tmp_ent[ent_idx*block_size:(ent_idx+1)*block_size] = 1
        noise = (np.random.randint(0, 10, size=tmp_ent.shape) < 4).astype(int)
        tmp_ent |= noise
        entities.append(tmp_ent)
    entities = np.vstack(entities)

    # assert entities are non-nested!!!
    assert np.all(np.argmax(entities @ entities.T, axis=1)
                  == np.array(range(entities.shape[0])))

    for _ in range(num_mentions):
        ent_idx = random.randint(0, num_entities-1)
        mention_labels.append(ent_idx)
        while True:
            ent_mask = (np.random.randint(0, 10, size=tmp_ent.shape) < 4).astype(int)
            sample_mention = ent_mask & entities[ent_idx]
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
