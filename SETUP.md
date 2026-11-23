# Run PDF Redline on your computer

Install [Git](https://git-scm.com/downloads) and [Python](https://www.python.org/downloads/). The project requires Python 3.11 or newer. The commands below use Python 3.12 on Windows. You can run a comparison in Local mode without Azure credentials.

## Clone and install

Open PowerShell on Windows or Terminal on macOS or Linux. Run these commands from the folder where you want to keep the project.

**Windows (PowerShell)**

```powershell
git clone https://github.com/kalyanvissamsetty/doc-tracer.git
cd doc-tracer
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ui,test]"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

**macOS or Linux**

```sh
git clone https://github.com/kalyanvissamsetty/doc-tracer.git
cd doc-tracer
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[ui,test]'
.venv/bin/python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

If GitHub asks you to sign in, use an account with access to the repository. Keep the terminal open and visit [the local app](http://127.0.0.1:8501) in your browser. If Streamlit reports that the port is busy, add `--server.port 8502` to the startup command and open `http://127.0.0.1:8502`. Press **Ctrl+C** in the terminal to stop the app.

## Try a comparison

1. Click **Run sample comparison**, then **Open review**. The built-in sample should show three proposed changes.
2. Select a change to inspect its original and revised regions. Accept, reject, or edit the proposed changes.
3. Click **Generate final PDF** and download the PDF and JSON change log. The final PDF includes accepted changes.
4. To use your own files, upload the original under **Original · V1** and the revision under **Revised · V2**. Keep **Mode** set to **Local**, then click **Run comparison**.

Local mode can use PDF text and visual differences. It may show raster regions for scanned tables. The 31-page BOM comparison shared in chat used an additional local OCR and visual review pass, so running the same pair through the app can produce different results.

## Run automated checks

With the terminal in the `doc-tracer` folder, run:

**Windows**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m redline.eval
```

**macOS or Linux**

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m redline.eval
```

The evaluator runs the two included synthetic PDF pairs. Live Azure tests are opt-in. Some Azure adapter tests also skip unless you install the `azure` extra.

## Optional Azure mode

Install `.[ui,test,azure]` in the same virtual environment. Copy `.env.example` to `.env` and configure your endpoints, deployments, credentials, token limits, and rates before choosing **Azure agent** in the UI. Keep credentials in `.env` or Streamlit secrets. See the [README](README.md) for cost settings and [DEPLOYMENT.md](DEPLOYMENT.md) for Entra ID, Blob Storage, Application Insights, and Docker.

The app keeps uploads, comparisons, and job history under `.redline/web/` by default. That directory and the local secret files are ignored by Git.
