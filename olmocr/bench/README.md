# olmOCR-Bench

We develop olmOCR-Bench in order to automatically and effectively evaluate document-level
parsing and OCR of various tools.

olmOCR-Bench works by testing various "facts" or "properties" about document pages at the PDF-level.
We choose PDFs directly, because PDFs do preserve some digital metadata and information which is helpful
and commonly available. Almost any other format can be converted to a PDF, but not the reverse.

## Property classes

- Text presence/absence
 - This task makes sure that a given small piece of text (ex. 1-3 sentence level) is present with high probability within
    a parsed document. It looks at documents with ambiguity around headers, footers, and other ambiguous content. Text still
    has a fuzzy matching allowed.
- Natural Reading Order
 - This task ensures that blocks of text which are present have a defined order relative to one another. For example,
  on a document that contains multiple news articles on one page, you'd want to see that the first sentence of the 
  first article appears after the heading of that article. But, you may be okay with swapping the order of those 
  two articles.
- Table Accuracy
 - Pages with tables get parsed out and are checked for accuracy on a direct row/column/title basis.
- Formula Accuracy
 - Extract formula from document, render it, and compare rendering using foundation model.

Table Format:
 - pdf_filename
 - Task ID
 - Type: text_presence, text_absense, reading_order, table
 - text_presence, text_absense: {text: str, fuzzy_threshold: float}
 - reading_order: {target_text_presence: task_id, appears_before: task_id, appears_after: task_id}
 - table: {table_index: int, needs to be fuzzy as well, ex. does row exist with column text X, does column exist with a row containing Y}
 - formula: TODO

## Creation

We sampled documents from the same source as olmocrmix. We run them through two models, and see which ones have biggest 
plain textual diffs, but still contain lots of good text, and aren't just tables/formula heavy for now.
Then, we will extract text presence/absense markers and verify using tinyhost UI. 
Write those to JSON. Maybe do some embedding and grouping to try to get lots of variation, at least when 
prioritizing manual review.

Later, we will repeat the same for tables and formulas.

Write the evalutor script which will output a nice templated tinyhostable results page.

## Running
We do not want to depend on a model having any specific format of its output.

Step 1. Download dataset with all pdfs (all will be single page) to /pdfs
Step 2. Run your extraction on it, point output to folder, ex. olmocr-v2_1/ where you expect pdf_page1.md for /pdfs/pdf_page1.pdf file
Step 3. Run the evaluation script
Step 4. Get results, and use tinyhost to view all failing examples

### Running existing scripts

```bash
pip install marker-pdf==1.5.4
python olmocr/bench/runners/run_marker.py olmocr/bench/sample_data/pdfs

pip install verovio torchvision
python olmocr/bench/runners/run_gotocr.py olmocr/bench/sample_data/pdfs

conda create -n MinerU python=3.10
conda activate MinerU
pip install -U magic-pdf[full]==1.1.0 --extra-index-url https://wheels.myhloli.com
pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py
python olmocr/bench/runners/run_mineru.py olmocr/bench/sample_data/pdfs
```
