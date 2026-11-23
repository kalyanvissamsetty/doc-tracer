# PDF Redline Agent

The PDF Redline Agent toolkit provides the foundation for comparing versions of a PDF. It probes PDF content, renders pages to local PNG artifacts, pairs pages by identifiers and content, detects changed raster regions, and writes annotated copies of revised PDFs from a change log. Azure wrappers provide chat with tools and images, plus cached Document Intelligence Read and Layout.

## Setup and tests

[Local setup and testing guide for Windows, macOS, and Linux](SETUP.md)

From the existing checkout, use Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

The default tests use generated PDFs and mock Azure responses. To exercise actual Azure services, configure the variables named in `.env.example` in a local `.env` or through secure environment settings. Configuration also accepts top-level Streamlit secrets. Environment variables take precedence, followed by Streamlit secrets and then `.env`.

`AZURE_OPENAI_API_VERSION` defaults to `v1`, which uses the OpenAI SDK with the Azure resource's `/openai/v1/` base URL. A dated API version selects the SDK's `AzureOpenAI` client. Use a resource HTTPS endpoint and your Azure deployment names. Document Intelligence uses API version `2024-11-30`. The live checks use generated PDF and image content and consume Azure quota.

```sh
REDLINE_LIVE_AZURE=1 python -m pytest -q tests/integration
```

## Milestone 2 structural tools

| Tool | Use and result | Limits |
| --- | --- | --- |
| `detect_table_grid` | Find the largest coherent ruled table in an upright grayscale or RGB render. Returns pixel rule positions or `None`. | Use Document Intelligence Layout for borderless, skewed, or complex tables. |
| `extract_table` | Map PDF words or a Document Intelligence page's OCR words into the grid. Returns cells with values and unrotated PDF boxes. | Supply OCR for image-only pages. OCR coordinates must match the render orientation. |
| `match_rows` | Align rows by unique item number, then by description and heat number together. Returns matched indices and unpaired rows. | Pass alternate keys for other header names. Missing and duplicate keys can leave rows unpaired. |
| `detect_stamps` | Propose colored ink regions while merging small gaps and dropping nested components. Returns pixel boxes. | Colored graphics can be candidates. Grayscale stamps and stamps that touch may need a different method. |
| `split_annotation_layer` | Produce the content render and a list of PDF review annotations, widgets, and links. | Flattened marks remain in page content. |
| `compare_embedded_images` | Compare embedded image occurrences by digest and placement. Returns moves, resizes, replacements, additions, and removals. | Replacements require sufficient box overlap. Use raster comparison for overlays or visual changes beyond image content and bounds. |
| `view_page` / `view_crop` | Read a stored render or a subregion as PNG bytes. Crop coordinates are displayed pixels. | Convert PDF point coordinates before cropping. |
| `normalize_value` | Normalize case, whitespace, equivalent unit spellings, and unambiguous dates. | Enable `numeric_ocr` only for numeric fields. Units of different magnitudes and ambiguous dates remain distinct. |
| `read_region_pair` | Interpret two PNG crops with prior deterministic evidence. Returns a schema-validated reading, the supplied evidence, and Azure usage. | The deployment must support vision and Structured Outputs. Pass `temperature=None` for models that require the default. The reader supplies no coordinates and does not discover candidate regions. |

## Milestone 3 command line

Run a local comparison with no Azure credentials:

```sh
python -m redline v1.pdf v2.pdf --offline -o redline.pdf
```

The command creates `redline.changes.json` beside the PDF and prints the path of the job's JSONL trace. Local mode uses deterministic discovery and checks the rendered markup. It does not use a model to interpret scan regions or judge semantic box placement. Ambiguous differences remain in the change log with low confidence.

For the Azure tool-calling workflow, configure the Azure variables from `.env.example` and omit `--offline`:

```sh
python -m redline v1.pdf v2.pdf -o redline.pdf --max-steps 60 --max-cost-usd 5
```

The agent deployment must support tool calling and image input. The vision deployment must support images and Chat Completions Structured Outputs. Set the `AGENT_*` and `VISION_*` rate and context variables for those deployments. The context bounds must cover each deployment's maximum input, including image and schema tokens. Use upper-bound input and output prices in USD per million tokens. Before each Azure request, the app reserves the cost of the full configured input context and the requested output cap. It reconciles against reported usage and retains the reservation when usage is unavailable. SDK retries are disabled for metered calls. Budget enforcement depends on accurate deployment bounds and prices.

Add `--docintel-layout` and set `DOCINTEL_LAYOUT_USD_PER_PAGE` to use metered Layout OCR on paired scanned ruled tables. Without OCR or a usable grid, the workflow compares raster regions. Scans are compared at render resolution, with proportional resizing when needed. As-built drawings use the registered scan workflow described below when applicable.

Every comparison probes both inputs, discovers deterministic candidates, and renders the latest markup for verification. In Azure mode, the agent receives the original V2 and marked page images and must approve each required page for the current revision. Rewrites invalidate prior approvals, and drafting stops after three markup revisions. A stopped or failed comparison leaves job artifacts for diagnosis and does not export the draft.

Traces record model and tool events, artifact IDs, spend estimates, and verification status. Job folders can contain document-derived images, OCR cache files, and change logs. Keep these folders local or restrict access as appropriate.

## Milestone 4 browser workflow

From the checkout, install the optional UI dependencies and start Streamlit:

```sh
python -m pip install -e '.[ui,test]'
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

1. In **New comparison**, upload V1 and V2 or run the built-in sample. Local mode needs no credentials. Select **Azure agent** to use the configured deployments and pricing bounds. The UI reads environment variables, `.env`, or top-level Streamlit secrets.
2. In **Review**, select a page and change to see the PDF, paired regions, evidence, and confidence. Accept, reject, or edit changes. An edit must be accepted before export. Bulk acceptance only accepts pending changes with at least 70% confidence. Generate the final PDF and download it with the review change log.
3. In **Jobs & traces**, inspect comparison status, time, estimated cost, tool inputs and outputs, and recorded thumbnails. Model call metadata is shown without raw prompts or credential values.

The prototype stores job states and review decisions in `.redline/web/jobs.sqlite3` and keeps each job's uploaded PDFs, artifacts, and JSONL trace under `.redline/web/`. Run one local Streamlit process per data directory. Running jobs are marked failed on restart. The app is for a trusted local workspace where the operator can access all jobs.

### Golden-set evaluation

Run **Run evaluation** in **Jobs & traces** or run:

```sh
python -m redline.eval
```

Each directory under `tests/golden` contains `v1.pdf`, `v2.pdf`, and `expected_changes.json`. Labels use one-based page numbers and unrotated PDF point boxes. Each scored detection matches one expected change by page, type, and bounding-box IoU of at least 0.3. Whole-page changes can use a null box, and deleted pages use the V1 page number. Type accuracy uses geometry-matched detections. Value accuracy scores explicitly labeled before and after values.

Use `--golden-dir` for another set. Use `--predictions-dir` to score existing logs at `<case>/changes.json` without rerunning comparisons. The default suite has two synthetic cases, so its scores are a smoke check.

## Milestone 5 drawing comparison and pressure validation

Select **As-built** in the app or run:

```sh
python -m redline v1.pdf v2.pdf --offline --document-type as-built -o drawing-redline.pdf
```

- `vector_diff` compares native PDF path primitives, stroke and fill styles, and text labels in PDF point coordinates. It ignores small geometry rounding within its point tolerance. Use it on native drawing exports.
- `align_images` estimates a V1-to-V2 transform from preview features using ORB or SIFT and RANSAC. Use it on scans with shared visible geometry. Weak matches produce a raster fallback marked for review.
- `compare_scanned_drawing` compares overlapping high-resolution tiles after registration, joins fragments across tile seams, and returns V1 and V2 region coordinates. It compares grayscale linework within the shared scan area. Severe crop differences, faint marks, and repetitive linework can require manual review. In Azure mode, the agent may read the original drawing crops. If metered Layout OCR is enabled, only the requested drawing crops are sent to OCR.

For pressure test reports, select **Pressure test** in the app or run:

```sh
python -m redline v1.pdf v2.pdf --offline --document-type pressure-test -o report-redline.pdf
```

Recognizable pressure test reports also trigger validation in Auto mode. A separate **Validation** section in Review reports the pressure range, duration, medium, chart attachment when required, date sequence, signature presence, and employee ID. The CLI writes the same results to `report-redline.validation.json`.

Configure approved limits and field aliases in `config/pressure_test_rules.yaml` or pass `--pressure-rules path.yaml` to the CLI. Blank pressure or duration limits leave those checks for review. Values need units such as `psi`, `MPa`, or `minutes`. Validation reads PDF text, form fields, and chart evidence in the revised document. Unreadable scan fields and missing limits can require manual review. Signature checks establish presence only.

## Coordinate and markup conventions

Page numbers are one-based. Change boxes are in unrotated PyMuPDF page points. Raster regions are in rendered pixel coordinates and require conversion before markup. Raw probe drawing counts can include annotation appearances. Page pairing uses order to resolve equally sized gaps where identifiers and content provide no match.

Markup preserves the original V2 page count and existing annotations. It places a legend in a clear page-edge strip where available, or in a redline comment on dense pages. Page deletions have no V2 page and are recorded in the change log. The local artifact store can hold document content in rendered images and OCR cache files.

## API references

- [Azure AI Foundry v1 REST contract](https://github.com/Azure/azure-rest-api-specs/blob/main/specification/ai/data-plane/OpenAI.v1/azure-v1-v1-generated.json)
- [Azure OpenAI API lifecycle and v1 client setup](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)
- [Azure OpenAI function calling](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/function-calling)
- [Azure OpenAI image inputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/gpt-with-vision)
- [Azure Document Intelligence Python SDK](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/documentintelligence/azure-ai-documentintelligence)

## Refine markup placement

Numbered tags look for clear space inside or beside each box. The renderer checks the underlying page and other marks before placing each tag. If it finds no clear spot, the number remains in the PDF annotation comment. Affected IDs are listed under `comment_only_tags` in the change log.

In **Review**, select a value change and choose **Snap box to detected cell**. A snap is available when one detected cell contains the evidence box. Boxes spanning cells or pages without a reliable grid stay available for review. The change log records the original and revised boxes, grid row and column, and actor under `geometry_edits`. Grid row and column numbers start at one and include the header row.

Snapping in Review refreshes the preview, clears prior acceptances on that V2 page, and makes the edited change require explicit acceptance. In Azure mode the agent can call `snap_to_cell` before rewriting and reviewing the new markup. Local comparison still performs deterministic render checks and relies on the reviewer to judge whether the proposed boxes are correct.

## Milestone 6 Azure deployment

The default local setup works without Azure integrations. Install `.[ui,azure]` for Blob archival, Application Insights tracing, and Entra authentication. See [Azure deployment and Docker instructions](DEPLOYMENT.md).
