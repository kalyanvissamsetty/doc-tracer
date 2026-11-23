"""Initialize optional Azure adapters once per application or CLI invocation."""
from contextlib import ExitStack
from pathlib import Path

from redline.config import CloudSettings, ConfigurationError
from redline.tracing import CompositeTracer, JsonlTracer, Tracer


class CloudServices:
    def __init__(self, config: CloudSettings):
        self.stack = ExitStack()
        self.store = None
        self.provider = None
        connection = config.azure_storage_connection_string
        account_url = config.azure_storage_account_url
        container = config.azure_storage_container
        if bool(container) != bool(connection or account_url) or (connection and account_url):
            raise ConfigurationError("Set AZURE_STORAGE_CONTAINER and exactly one Blob connection method.")
        try:
            if container:
                from azure.storage.blob import ContainerClient

                from redline.storage.blob import BlobStorage

                if connection:
                    client = ContainerClient.from_connection_string(connection.get_secret_value(), container)
                else:
                    credential = entra_credential()
                    self.stack.callback(credential.close)
                    client = ContainerClient(account_url, container, credential=credential)
                self.store = BlobStorage(client)
                self.stack.callback(self.store.close)
            if config.applicationinsights_connection_string:
                from redline.tracing.insights import create_provider

                self.provider = create_provider(config.applicationinsights_connection_string.get_secret_value())
                self.stack.callback(self.provider.shutdown)
        except ImportError:
            self.close()
            raise ConfigurationError("Install the Azure integrations with pip install '.[azure]'.") from None
        except ValueError:
            self.close()
            raise ConfigurationError("Check the Azure Blob and Application Insights configuration.") from None

    def tracer(self, path: Path, job_id: str) -> Tracer:
        local = JsonlTracer(path)
        if self.provider is None:
            return local
        from redline.tracing.insights import InsightsTracer

        return CompositeTracer(local, InsightsTracer(self.provider.get_tracer("redline"), job_id))

    def close(self):
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def entra_credential():
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise ConfigurationError("Install the Azure integrations to use Entra ID authentication.") from None
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)
