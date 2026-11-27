"""Run scoped calibration benchmark with measured synthetic CPU contention.

CPU workers are child processes stopped in a finally block. This measures one
synthetic condition, not the user's normal workload or a hard-real-time system.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--profile',choices=('normal','noisy'),required=True)
    parser.add_argument('--load-workers',type=int,default=0)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.load_workers < 0 or args.load_workers > min(4,os.cpu_count() or 1):
        parser.error('load-workers must be between 0 and min(4, logical CPUs)')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    workers=[]
    start=time.perf_counter()
    result=None
    try:
        for _ in range(args.load_workers):
            # Fixed-size integer arithmetic keeps a core active without allocating memory.
            code='x=1\nwhile True:\n x=(x*1664525+1013904223)&0xffffffff'
            flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
            # Windows venv launcher spawns another process; use the base interpreter
            # so the measured and terminated process is the actual CPU worker.
            worker_python=getattr(sys,'_base_executable',None) or sys.executable
            child=subprocess.Popen([worker_python,'-c',code],creationflags=flags,
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            workers.append(child)
        result=subprocess.run([sys.executable,'-B','-m','experiments.scoped_calibration_benchmark',
                 '--seed',str(args.seed),'--profile',args.profile,'--output-dir',str(args.output_dir)],check=False)
    finally:
        elapsed=time.perf_counter()-start
        usage=[]
        for child in workers:
            try:
                proc=psutil.Process(child.pid)
                t=proc.cpu_times();usage.append(t.user+t.system)
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                usage.append(None)
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
        data=dict(seed=args.seed,profile=args.profile,load_workers=args.load_workers,
                  logical_cpus=os.cpu_count(),wall_seconds=elapsed,worker_cpu_seconds=usage,
                  child_exit_code=None if result is None else result.returncode)
        (args.output_dir/'load_metadata.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    if args.load_workers and (any(v is None or v <= 0 for v in usage)):
        raise RuntimeError('CPU load was not observed in all worker processes; discard this run')
    if result is None or result.returncode:
        raise SystemExit(1 if result is None else result.returncode)

if __name__=='__main__': main()
