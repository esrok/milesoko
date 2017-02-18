from __future__ import print_function

from contextlib import contextmanager
from datetime import datetime, timedelta
from logging import getLogger
from time import sleep

import boto3
from paramiko import SSHClient, AutoAddPolicy, MissingHostKeyPolicy, RejectPolicy
from paramiko.ssh_exception import SSHException, NoValidConnectionsError


logger = getLogger('aws-spot-runner')


class AWSInstancePolicy(MissingHostKeyPolicy):
    GENERATION_STR = 'Generating public/private'
    FINGERPRINT_STR = 'The key fingerprint is:'
    FINGERPRINT_OFFSET = 4

    key_type_mapping = {
        'rsa': 'ssh-rsa',
        'dsa': 'ssh-dss',
        'ecdsa': 'ssh-ecdsa',
    }

    def _parse_fingerprints(self, console_output):
        lines = console_output.split('\n')
        for index, line in enumerate(lines):
            fingerprint_line_index = index + self.FINGERPRINT_OFFSET
            if fingerprint_line_index >= len(lines):
                break

            if line.startswith(self.GENERATION_STR):
                pre_fingerprint_line = lines[fingerprint_line_index - 1]
                fingerprint_line = lines[fingerprint_line_index]
                if not pre_fingerprint_line.startswith(self.FINGERPRINT_STR):
                    continue

                for key_type, key_mapping in self.key_type_mapping.iteritems():
                    if key_type in line:
                        result_mapping = key_mapping
                        break
                else:
                    # key type mapping was not found
                    logger.info('No key type was found in "%s"', line)
                    continue

                key_fingerprint = fingerprint_line.split(' ')[0]
                key_fingerprint = ''.join(key_fingerprint.split(':'))
                logger.info('Found %s key fingerprint %s', result_mapping, key_fingerprint)
                yield result_mapping, key_fingerprint

    def __init__(self, instance):
        self._instance = instance
        self._hostname = instance.public_dns_name
        self._ip_address = instance.public_ip_address
        self._fingerprints = None
        self._reject = RejectPolicy()
        self._accept = AutoAddPolicy()

    def _verify(self, key):
        hex_key_fingerprint = key.get_fingerprint().encode('hex')
        name = key.get_name()
        if name in self._fingerprints:
            return self._fingerprints[name] == hex_key_fingerprint
        return False

    def missing_host_key(self, client, hostname, key):
        if self._fingerprints is None:
            self._fingerprints = dict(self._parse_fingerprints(self._instance.console_output()['Output']))
        result = hostname in (self._hostname, self._ip_address)
        result = result and self._verify(key)
        if not result:
            logger.error('Key verification failed for %s', hostname)
            return self._reject.missing_host_key(client, hostname, key)
        return self._accept.missing_host_key(client, hostname, key)


class AWSSpotInstanceRunner(object):
    DEFAULT_TIMEOUT = timedelta(minutes=5)
    DEFAULT_WAIT_STEP = 15  # seconds

    EC2_USERNAME = 'ec2-user'

    STATUS_CHECK_OK = 'ok'
    INSTANCE_STATE_RUNNING = 'running'
    INSTANCE_STATE_PENDING = 'pending'
    REQUEST_STATE_ACTIVE = 'active'
    REQUEST_STATE_OPEN = 'open'
    REQUEST_STATUS_FULFILLED = 'fulfilled'


    def __init__(self, spot_price, launch_spec, volume=None, client_token=None, username=EC2_USERNAME):
        self._ssh_client = SSHClient()
        self._instance = self._instance_id = None
        self._spot_instance_request_id = None
        self._state = None
        self._status = None
        self._client = boto3.client('ec2')
        self._ec2 = boto3.resource('ec2')

        self._spot_price = spot_price
        self._launch_spec = launch_spec
        self._volume = volume
        self._client_token = client_token
        self._username = username

    def _request_spot_instance(self, valid_until=None, dry_run=False, ):
        now = datetime.utcnow()
        valid_until = now + timedelta(minutes=5)
        response = self._client.request_spot_instances(
            DryRun=dry_run,
            InstanceCount=1,
            Type='one-time',
            SpotPrice=self._spot_price,
            LaunchSpecification=self._launch_spec,
            ValidUntil=valid_until,
        )
        spot_instance_request = response['SpotInstanceRequests'][0]
        self._spot_instance_request_id = spot_instance_request['SpotInstanceRequestId']
        logger.info('Got spot instance request with id %s', self._spot_instance_request_id)
        self._state = spot_instance_request['State']
        self._status = spot_instance_request['Status']['Code']
        logger.info('Current spot instance request state %s %s', self._state, self._status)
        return self._spot_instance_request_id

    def _wait_until_fulfilled(self, timeout=DEFAULT_TIMEOUT, step=DEFAULT_WAIT_STEP):
        now = start = datetime.utcnow()
        while now < start + timeout:
            response = self._client.describe_spot_instance_requests(
                SpotInstanceRequestIds=[self._spot_instance_request_id],
            )
            for description in response['SpotInstanceRequests']:
                if description['SpotInstanceRequestId'] == self._spot_instance_request_id:
                    self._state = description['State']
                    self._status = description['Status']['Code']
                    logger.info(
                        'Current spot instance request state %s %s',
                        self._state,
                        self._status,
                    )
                    if self._state == self.REQUEST_STATE_ACTIVE and \
                            self._status == self.REQUEST_STATUS_FULFILLED:
                        self._instance_id = description['InstanceId']
                        self._instance = self._ec2.Instance(self._instance_id)
                        return self._instance
                    elif self._state == 'open':
                        if self._status == 'price-too-low':
                            logger.info('Price is too low')
                    else:
                        # TODO: check other requests states
                        pass
                    break
            now = datetime.utcnow()
            sleep(step)
        return None

    def _check_reachability(self, instance_status):
        for check_name in ('InstanceStatus', 'SystemStatus'):
            check = instance_status[check_name]
            if check['Status'] != self.STATUS_CHECK_OK:
                logger.info('%s check is %s', check_name, check['Status'])
                return False

        return True

    def _wait_until_running(self, timeout=DEFAULT_TIMEOUT, step=DEFAULT_WAIT_STEP, wait_reachable=True):
        now = start = datetime.utcnow()
        while now < start + timeout:
            response = self._client.describe_instance_status(InstanceIds=[self._instance_id])
            if len(response['InstanceStatuses']) == 1:
                instance_status = response['InstanceStatuses'][0]
                instance_state = instance_status['InstanceState']['Name']
                logger.info('Instance now in state %s', instance_state)
                if instance_state == self.INSTANCE_STATE_RUNNING:
                    if not wait_reachable or self._check_reachability(instance_status):
                        logger.info('Instance is running and reachable')
                        return self.INSTANCE_STATE_RUNNING
                elif instance_state != self.INSTANCE_STATE_PENDING:
                    logger.warn('Unexpected instance state %s', instance_state)
            sleep(step)

    @property
    def ssh_creds(self):
        assert self._instance is not None, \
            'AWS instances must be running to get ssh connection'
        return '{username}@{hostname}'.format(
            username=self._username,
            hostname=self._instance.public_dns_name,
        )

    def save_ssh_keys(self, filename):
        return self._ssh_client.save_host_keys(filename)

    def _ssh_connect(self, timeout=DEFAULT_TIMEOUT, step=DEFAULT_WAIT_STEP):
        transport = self._ssh_client.get_transport()
        if transport is None or not transport.active:
            self._ssh_client.set_missing_host_key_policy(AWSInstancePolicy(self._instance))
            self._ssh_client.load_system_host_keys()
            now = start = datetime.utcnow()
            while now < start + timeout:
                try:
                    self._ssh_client.connect(
                        self._instance.public_dns_name,
                        username=self._username,
                    )
                    logger.info(
                        'Successfully connected at %s@%s',
                        self._username,
                        self._instance.public_dns_name,
                    )
                    break
                except (SSHException, NoValidConnectionsError):
                    logger.warn(
                        'Got SSH exception, while connecting to %s',
                        self._instance.public_dns_name,
                        exc_info=True,
                    )
                sleep(step)

    def exec_command(self, command, *args, **kwargs):
        self._ssh_connect()
        return self._ssh_client.exec_command(command, *args, **kwargs)

    def _attach_volume(self, device='/dev/xvdh'):
        response = self._client.attach_volume(
            VolumeId=self._volume,
            InstanceId=self._instance_id,
            Device=device,
        )
        logger.info(
            'Attaching volume %s as device %s',
            self._volume,
            response['Device'],
        )

    def _launch(self, valid_until=None, dry_run=False, wait_reachable=True):
        self._request_spot_instance(valid_until, dry_run)
        self._wait_until_fulfilled()
        self._wait_until_running(wait_reachable=wait_reachable)
        self._ssh_connect()
        if self._volume is not None:
            self._attach_volume()

    @contextmanager
    def launch(self, valid_until=None, dry_run=False, wait_reachable=True):
        try:
            self._launch(valid_until, dry_run, wait_reachable)
            yield
        finally:
            self._close()

    def _close(self):
        if self._instance is not None:
            self._instance.terminate()
        self._ssh_client.close()
        # TODO: cancel open spot instance requests
        # if self._state == self.REQUEST_STATE_OPEN:
        #     self._cancel_spot_request()
