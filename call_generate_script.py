import os
import sys

jobs_file = sys.argv[-1]
jobs_list = open(jobs_file, 'r').readlines()

for job in jobs_list:
    os.system(job)