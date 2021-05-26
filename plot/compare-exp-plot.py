import os
import sys
import glob
import re
import json
from collections import defaultdict

from IPython import embed



def parse_log(fname, parsed_data):

    method_re = re.compile('python src/(\S*).py')
    dataset_re = re.compile('--data_file=(\S*).pkl')
    round_metrics_re = re.compile('metrics: (\{.*\})')

    method = None
    dataset = None
    round_metrics = []

    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            method_re_match = method_re.search(line)
            dataset_re_match = dataset_re.search(line)
            round_metrics_re_match = round_metrics_re.search(line)

            if method_re_match is not None:
                if method is not None:
                    assert method == method_re_match.group(1)
                else:
                    method = method_re_match.group(1)

            if dataset_re_match is not None:
                if dataset is not None:
                    assert dataset == dataset_re_match.group(1)
                else:
                    dataset = dataset_re_match.group(1)

            if round_metrics_re_match is not None:
                round_metrics.append(
                    json.loads(
                        round_metrics_re_match.group(1).replace("'", '"')
                    )
                )

    for rnd_mets in round_metrics:
        parsed_data['dataset'].append(dataset)
        parsed_data['method'].append(method)
        parsed_data['#constraints'].append(rnd_mets['# constraints'])
        parsed_data['dp'].append(rnd_mets['dp'])
        parsed_data['rand'].append(rnd_mets['adj_rand_idx'])
        parsed_data['nmi'].append(rnd_mets['adj_mut_info'])


log_fnames = list(glob.glob('../experiments/exp[0-9]*/console.log'))

parsed_data = defaultdict(list)
for fname in log_fnames:
    parse_log(fname, parsed_data)

parsed_data = dict(parsed_data)

embed()
exit()
