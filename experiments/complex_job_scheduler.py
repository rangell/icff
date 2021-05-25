import itertools
import collections
import glob
import os
import datetime
import subprocess
import string
import sys


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def get_run_id():
    filename = 'expts.txt'
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts) / 5
    return run_id

top_details = 'Initial runs of icff over all trimmed datasets'

hyperparameters = [
    [('seed',), ['27']],
    [('data_file',), ['r8-test-stemmed.dataset.trimmed.pkl',
                      'r52-test-stemmed.dataset.trimmed.pkl',
                      'cade-test-stemmed.dataset.trimmed.pkl',
                      'webkb-test-stemmed.dataset.trimmed.pkl',
                      '20ng-test-stemmed.dataset.trimmed.pkl']],
]

run_id = int(get_run_id())
key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []

for combo in combinations:
    with open("icff_grid_template.sh", 'r') as f:
        train_script = f.read()

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    # for k, v in other_dependencies.items():
    #     combo[k] = v(combo)

    od = collections.OrderedDict(sorted(combo.items()))
    lower_details = ""
    for k, v in od.items():
        lower_details += "%s = %s, " % (k, str(v))
    # removing last comma and space
    lower_details = lower_details[:-2]

    combo["top_details"] = top_details
    combo["lower_details"] = lower_details
    combo["job_id"] = run_id
    print("Scheduling Job #%d" % run_id)

    for k, v in combo.items():
        if "{%s}" % k in train_script:
            train_script = train_script.replace("{%s}" % k, str(v))

    train_script += "\n"

    # Write schedule script
    script_name = 'slurm-gen-scripts/exp%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(train_script)

    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name + "\n" + \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
        top_details + "\n" + \
        lower_details + "\n\n"
    with open("expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1


# schedule jobs
for script in scripts:
    command = "sbatch %s" % script
    print(subprocess.check_output(command, shell=True))
