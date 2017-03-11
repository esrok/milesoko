import logging
import logging.config
import select

import yaml

from aws_runner import AWSSpotInstanceRunner


with open('config/amazon-deep-learning.example.yml', 'r') as f:
    config = yaml.load(f)


with open('config/logging.example.yml', 'r') as f:
    logging_config = yaml.load(f)
    logging.config.dictConfig(logging_config)


logger = logging.getLogger('spot-training-example')


def select_exec_command(instance, command, timeout=1.0):
    channel = instance._ssh_client.get_transport().open_session()
    channel.set_combine_stderr(True)
    channel.exec_command(command)
    while not channel.exit_status_ready():
        rlist, wlist, xlist = select.select([channel], [], [], timeout)
        if rlist:
            yield channel.recv(1024)


def exec_command(instance, command):
    # this method is necessary for long-running commands like `tail -f`
    output = ''.join(select_exec_command(instance, command))
    logger.info(output)


instance = AWSSpotInstanceRunner(**config)
with instance.launch(wait_reachable=True):
    ssh_creds = instance.ssh_creds
    ssh_hosts_filename = '/tmp/known_hosts'

    logger.info('Creating mounting point')
    exec_command(instance, 'mkdir /home/ec2-user/storage')
    logger.info('Mounting volume')
    exec_command(instance, 'sudo mount /dev/xvdh /home/ec2-user/storage')

    logger.info('Storing ssh keys in %s', ssh_hosts_filename)
    instance.save_ssh_keys(ssh_hosts_filename)

    logger.info('Syncing sources to %s', ssh_creds)
    # sync sources

    logger.info('Running init scripts')
    # run init scripts

    logger.info('Running training')
    # run training
