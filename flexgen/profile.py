import argparse
import configparser
import os
import sys

import profile_bandwidth
import profile_matmul

parser = argparse.ArgumentParser()
parser.add_argument("--offload-path", type=str, default="~/flexgen_offload_dir/tmp.npy")
args = parser.parse_args()

bdws = {}
offload_path = os.path.expanduser(args.offload_path)
for record in profile_bandwidth.profile_bandwidth(offload_path):
    key = record[0][0] + '2' + record[1][0]
    val = record[3]
    bdws[key] = min(bdws[key], val) if key in bdws else val

config = configparser.ConfigParser()
config['bandwidth'] = bdws

flopses = {}
for record in profile_matmul.bench_matmul():
    key = record[0]
    val = record[3]
    flopses[key] = min(flopses[key], val) if key in flopses else val

config['matmul'] = flopses

config.write(sys.stdout)
