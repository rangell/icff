import copy
from functools import reduce


class TreeNode(object):

    def __init__(self, uid, raw_rep, transformed_rep=None, children=[]):
        self.uid = uid
        self.raw_rep = copy.deepcopy(raw_rep)
        self.transformed_rep = (
                copy.deepcopy(raw_rep) if transformed_rep is None
                    else copy.deepcopy(transformed_rep)
        )
        self.children = children
        self.parent = None

        for child in self.children:
            child.parent = self

    def get_leaves(self):
        if len(self.children) == 0:
            return [self]
        return reduce(
            lambda a, b : a + b,
            [c.get_leaves() for c in self.children]
        )

    def __repr__(self):
        children_uids = str([c.uid for c in self.children])
        parent_uid = self.parent.uid if self.parent is not None else 'None'

        return "{" + "\n".join([
            f"{'uid:':<25}{self.uid:<40}",
            f" {'raw_rep:':<25}{str(self.raw_rep):<40}",
            f" {'transformed_rep:':<25}{str(self.transformed_rep):<40}",
            f" {'children:':<25}{children_uids:<40}",
            f" {'parent:':<25}{parent_uid:<40}"
        ]) + "}"
