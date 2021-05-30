import os
import sys
import glob
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython import embed


sns.set_theme(style="darkgrid")


uniq = lambda x : list(set(x))


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

dp_df = pd.DataFrame(
    parsed_data, columns=['dataset', 'method', '#constraints', 'dp']
)
#dp_df = pd.DataFrame(
#    parsed_data, columns=['dataset', 'method', '#constraints', 'nmi']
#)

r8_dp_df = dp_df.query("dataset == 'r8-test-stemmed.dataset'")
#r52_dp_df = dp_df.query("dataset == 'r52-test-stemmed.dataset'")
#webkb_dp_df = dp_df.query("dataset == 'webkb-test-stemmed.dataset'")
#ng20_dp_df = dp_df.query("dataset == '20ng-test-stemmed.dataset'")
#jones_s_dp_df = dp_df.query("dataset =='rexa-jones_s'")
#allen_d_dp_df = dp_df.query("dataset =='rexa-allen_d'")
#young_s_dp_df = dp_df.query("dataset =='rexa-young_s'")
#moore_a_dp_df = dp_df.query("dataset =='rexa-moore_a'")
#robinson_h_dp_df = dp_df.query("dataset =='rexa-robinson_h'")
#mcguire_j_dp_df = dp_df.query("dataset =='rexa-mcguire_j'")


sns.lineplot(data=r8_dp_df, x="#constraints",  y="dp", hue="method")
#sns.lineplot(data=ng20_dp_df, x="#constraints",  y="dp", hue="method")
#sns.lineplot(data=jones_s_dp_df, x="#constraints",  y="dp", hue="method")
#sns.lineplot(data=allen_d_dp_df, x="#constraints",  y="dp", hue="method")
#sns.lineplot(data=young_s_dp_df, x="#constraints",  y="dp", hue="method")
#sns.lineplot(data=moore_a_dp_df, x="#constraints",  y="dp", hue="method")
#sns.lineplot(data=robinson_h_dp_df, x="#constraints",  y="dp", hue="method")
#sns.lineplot(data=mcguire_j_dp_df, x="#constraints",  y="dp", hue="method")

plt.show()
