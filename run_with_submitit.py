# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import argparse
import os, sys
import uuid
from pathlib import Path

import main as detection
import submitit


def parse_args():
    detection_parser = detection.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for detection", parents=[detection_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=60, type=int, help="Duration of the job")
    parser.add_argument("--cpus_per_task", default=16, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--job_name", type=str, help="Job name.")
    parser.add_argument("--qos", type=str, default=None, help="specify preemptive QOS.")
    parser.add_argument("--requeue", action='store_true', help="job requeue if preempted.")
    parser.add_argument("--mail_type", type=str, default='ALL', help=" send email when job begins, ends, fails or preempted.")
    parser.add_argument("--mail_user", type=str, default='', help=" email address.")
    # refer to https://slurm.schedmd.com/sbatch.html & \
    # https://github.com/facebookincubator/submitit/blob/11d8f87f785669e8a01aa9773a107f9180a63b09/submitit/slurm/slurm.py \
    # for more details about parameters of slurm.
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/comp_robot").is_dir():
        p = Path(f"/comp_robot/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        self._setup_gpu_args()
        detection.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.job_dir
        self.args.output_dir = str(self.args.output_dir).replace("%j", str(job_env.job_id))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")



def main():
    args = parse_args()
    args.commad_txt = "Command: "+' '.join(sys.argv)
    if args.job_dir == "":
        raise ValueError("You must set job_dir mannually.")

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    qos = args.qos

    additional_parameters = {
        'mail-user': args.mail_user,
        'mail-type': args.mail_type,
    }
    if args.requeue:
        additional_parameters['requeue'] = args.requeue


    executor.update_parameters(
        mem_gb=50 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=16,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        qos=qos,
        slurm_additional_parameters=additional_parameters
    )

    executor.update_parameters(name=args.job_name)
    args.dist_url = get_init_file().as_uri()

    # run and submit
    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
