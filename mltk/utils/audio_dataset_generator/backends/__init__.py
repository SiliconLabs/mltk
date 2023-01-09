

from .aws_backend import AwsBackend
from .gcp_backend import GcpBackend
from .azure_backend import AzureBackend


BACKENDS = dict()

BACKENDS['aws'] = AwsBackend
BACKENDS['gcp'] = GcpBackend
BACKENDS['azure'] = AzureBackend
