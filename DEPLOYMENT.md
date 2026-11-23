# Deploy PDF Redline to Azure Container Apps

The app container runs Streamlit as UID 10001 and listens on port 8501. Start with a private, single-replica Container App. The prototype shares each job with everyone who can access the app, so use internal ingress and your organization’s access controls. Review state is stored in SQLite and designed for one application process.

## Build and verify locally

```sh
docker build -t pdf-redline-agent:local .
docker run --rm -p 127.0.0.1:8501:8501 -v redline-data:/data pdf-redline-agent:local
```

Open the app at port 8501 and select **Run sample comparison**. A proxy that re-signs HTTPS traffic can supply its trusted CA bundle at build time with `--secret id=pip_ca,src=/path/to/ca-bundle.pem`. The build keeps TLS verification enabled and leaves the bundle out of the image.

## Configure Azure

1. Push the app image to your Azure Container Registry and grant the Container App’s image-pull identity access. Create a Container Apps environment and deploy the image with internal ingress targeting port 8501. Set both replica bounds to one. Use a deployment strategy that prevents two revisions from sharing the same SQLite working directory.
2. Configure startup and readiness probes on `/_stcore/health`. The image defaults to `REDLINE_DATA_DIR=/data`. Use writable local storage for the SQLite database and job working files. With replica-local storage, replacing the replica loses local job and review state. Blob archives persist independently. A durable history requires a storage design that supports SQLite locking or an external job database.
3. Set the Azure OpenAI and Document Intelligence endpoints and deployment names from `.env.example`. Configure the two model token bounds and prices, plus the OCR rate if OCR is enabled. Put keys and connection strings in Container Apps secrets or Key Vault references. Set `AZURE_AUTH_MODE=entra` instead for a managed identity with **Cognitive Services OpenAI User** on the OpenAI resource and **Cognitive Services User** on Document Intelligence. The identity must also be allowed to call the configured deployments. System-assigned identity is the default. For a user-assigned identity, set `AZURE_CLIENT_ID`.
4. To retain completed outputs, create a private Blob container and set `AZURE_STORAGE_CONTAINER`. Choose `AZURE_STORAGE_ACCOUNT_URL` with **Storage Blob Data Contributor** for the managed identity, or supply `AZURE_STORAGE_CONNECTION_STRING` from a secret. The app archives both input PDFs, the verified draft, change log, trace, any validation report, and the PDFs and logs of exported review revisions. A manifest is written last for each archive. Intermediate renders and in-progress review decisions stay in the local working directory.
5. Set `APPLICATIONINSIGHTS_CONNECTION_STRING` to enable the OpenTelemetry exporter. It forwards allowlisted job and tool metadata such as timing and token counts. Prompts, document text, artifact IDs, and credentials stay out of exported spans. Local JSONL remains available for each job.
6. Run a local-mode sample after rollout, then run an Azure-mode sample using the configured deployment. Verify the two downloadable outputs, the Blob manifest if enabled, and `redline.*` events in Application Insights. Test the as-built and pressure workflows with representative PDFs before relying on production results.

## Optional Python fallback

The hosted Streamlit app uses the registered PDF tools. The optional Python fallback runs on a separately managed Docker host with cgroup v2 enforcement:

```sh
docker build -f Dockerfile.sandbox -t redline-python-sandbox:local .
python -m redline v1.pdf v2.pdf --allow-sandbox -o redline.pdf
```

Each enabled call starts a fresh nonroot container with no networking or host mounts, a read-only root, a bounded temporary filesystem, and CPU, memory, process, output, and time limits. It refuses execution if resource limits are unavailable. The fallback handles pure calculations and receives no uploaded files, host credentials, or Docker socket. Keep the Docker daemon on the trusted host and leave the fallback disabled in Azure Container Apps.
