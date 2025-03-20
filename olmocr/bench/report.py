import glob
import os
from typing import Dict, List, Tuple

from olmocr.data.renderpdf import render_pdf_to_base64webp

from .tests import BasePDFTest


def generate_html_report(
    test_results_by_candidate: Dict[str, Dict[str, Dict[int, List[Tuple[BasePDFTest, bool, str]]]]], pdf_folder: str, output_file: str
) -> None:
    """
    Generate an enhanced HTML report of test results.

    Args:
        test_results_by_candidate: Dictionary mapping candidate name to dictionary mapping PDF name to dictionary
                                  mapping page number to list of (test, passed, explanation) tuples.
        pdf_folder: Path to the folder containing PDF files.
        output_file: Path to the output HTML file.
    """
    candidates = list(test_results_by_candidate.keys())

    # Create HTML report
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OLMOCR Bench Test Report</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/auto-render.min.js"></script>
    <style>
        :root {
            --primary-color: #4a86e8;
            --success-color: #34a853;
            --error-color: #ea4335;
            --warning-color: #fbbc05;
            --light-gray: #f5f5f5;
            --medium-gray: #e0e0e0;
            --dark-gray: #757575;
            --border-color: #ddd;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
            line-height: 1.6;
            background: #f9f9f9;
        }
        
        header {
            background: var(--primary-color);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        header h1 {
            margin: 0;
            font-weight: 400;
        }
        
        .container {
            max-width: 2200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .pdf-container {
            display: flex;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            background: white;
            overflow: hidden;
        }
        
        .pdf-view {
            padding: 20px;
            border-right: 1px solid var(--border-color);
            background: white;
        }
        
        .test-results {
            width: 700px;
            padding: 20px;
            overflow-y: auto;
            max-height: 800px;
        }
        
        .pdf-image {
            max-width: 100%;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }
        
        h2 {
            margin-top: 30px;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 10px;
        }
        
        h3 {
            margin-top: 20px;
            color: #444;
            font-weight: 500;
        }
        
        .test {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #ccc;
            transition: all 0.3s ease;
            background-color: white;
        }
        
        .test:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        
        .test.pass {
            background-color: #f0f8f0;
            border-left-color: var(--success-color);
        }
        
        .test.fail {
            background-color: #fff0f0;
            border-left-color: var(--error-color);
        }
        
        .test-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .test-id {
            font-weight: bold;
            font-size: 1.1em;
            color: #333;
        }
        
        .test-type {
            background: var(--medium-gray);
            color: #333;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .test-details {
            margin: 10px 0;
            font-size: 0.95em;
        }
        
        .explanation {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.03);
            border-radius: 5px;
            font-size: 0.9em;
            color: var(--dark-gray);
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
            background: white;
            border-radius: 8px 8px 0 0;
            overflow-x: auto;
            white-space: nowrap;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .tab {
            padding: 12px 24px;
            cursor: pointer;
            border: 1px solid transparent;
            font-weight: 500;
            color: var(--dark-gray);
            transition: all 0.3s ease;
        }
        
        .tab:hover {
            background: var(--light-gray);
            color: var(--primary-color);
        }
        
        .tab.active {
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            background: white;
        }
        
        .candidate-content {
            display: none;
        }
        
        .candidate-content.active {
            display: block;
        }
        
        .math {
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        
        .status-badge {
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        .status-badge.pass {
            background: var(--success-color);
            color: white;
        }
        
        .status-badge.fail {
            background: var(--error-color);
            color: white;
        }
        
        .page-tabs {
            display: flex;
            margin-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            overflow-x: auto;
        }
        
        .page-tab {
            padding: 8px 16px;
            cursor: pointer;
            border: 1px solid transparent;
            font-size: 0.9em;
        }
        
        .page-tab.active {
            border: 1px solid var(--border-color);
            border-bottom-color: white;
            margin-bottom: -1px;
            background: white;
        }
        
        .page-content {
            display: none;
        }
        
        .page-content.active {
            display: block;
        }
        
        .md-content {
            background: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #eee;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
            margin: 15px 0;
        }
        
        .md-toggle {
            background: var(--light-gray);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .md-toggle:hover {
            background: var(--medium-gray);
        }
        
        .summary {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .summary-table th, .summary-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .summary-table th {
            background-color: var(--light-gray);
            font-weight: 500;
        }
        
        .summary-table tr:hover {
            background-color: var(--light-gray);
        }
        
        @media (max-width: 1024px) {
            .pdf-container {
                flex-direction: column;
            }
            .pdf-view, .test-results {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>OLMOCR Bench Test Report</h1>
        </div>
    </header>
    
    <div class="container">
        <div class="summary">
            <h2>Summary</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Candidate</th>
                        <th>Pass Rate</th>
                        <th>Passed Tests</th>
                        <th>Failed Tests</th>
                        <th>Total Tests</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Calculate summary statistics for each candidate
    for candidate in candidates:
        total_tests = 0
        passed_tests = 0

        for pdf_name in test_results_by_candidate[candidate]:
            for page in test_results_by_candidate[candidate][pdf_name]:
                for test, passed, _ in test_results_by_candidate[candidate][pdf_name][page]:
                    total_tests += 1
                    if passed:
                        passed_tests += 1

        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        html += f"""                <tr>
                    <td>{candidate}</td>
                    <td>{pass_rate:.1f}%</td>
                    <td>{passed_tests}</td>
                    <td>{failed_tests}</td>
                    <td>{total_tests}</td>
                </tr>
"""

    html += """            </tbody>
            </table>
        </div>
        
        <div class="tabs">
"""

    # Create tabs for each candidate
    for i, candidate in enumerate(candidates):
        html += f"""        <div class="tab{' active' if i == 0 else ''}" onclick="switchTab('{candidate}')">{candidate}</div>
"""

    html += """    </div>
"""

    # Create content for each candidate
    for i, candidate in enumerate(candidates):
        html += f"""    <div id="{candidate}" class="candidate-content{' active' if i == 0 else ''}">
"""

        all_pdfs = set()
        for c in test_results_by_candidate.values():
            all_pdfs.update(c.keys())

        for pdf_name in sorted(all_pdfs):
            if pdf_name not in test_results_by_candidate[candidate]:
                continue

            html += f"""        <h2>PDF: {pdf_name}</h2>
"""

            # Create page tabs for this PDF
            pages = sorted(test_results_by_candidate[candidate][pdf_name].keys())
            html += f"""        <div class="page-tabs" id="{candidate}_{pdf_name.replace('.', '_')}_tabs">
"""
            for j, page in enumerate(pages):
                html += f"""            <div class="page-tab{' active' if j == 0 else ''}" 
                onclick="switchPageTab('{candidate}_{pdf_name.replace('.', '_')}', {page})">
                Page {page}
            </div>
"""
            html += """        </div>
"""

            # Create content for each page
            for j, page in enumerate(pages):
                html += f"""        <div id="{candidate}_{pdf_name.replace('.', '_')}_{page}" 
                class="page-content{' active' if j == 0 else ''}">
            <div class="pdf-container">
                <div class="pdf-view">
                    <h3>PDF Page {page}</h3>
"""

                # Convert PDF page to embedded image
                pdf_path = os.path.join(pdf_folder, pdf_name)
                try:
                    image_data = render_pdf_to_base64webp(pdf_path, page, 2048)
                    html += f"""                    <img class="pdf-image" alt="PDF Page {page}" src="data:image/webp;base64,{image_data}" />
"""
                except Exception as e:
                    html += f"""                    <div class="error">Error rendering PDF: {str(e)}</div>
"""

                html += """                </div>
                <div class="test-results">
                    <h3>Tests for Page {page}</h3>
"""

                # Get the Markdown file and content for this page
                md_file_path = None
                md_content = None
                try:
                    md_base = os.path.splitext(pdf_name)[0]
                    md_files = list(glob.glob(os.path.join(os.path.dirname(pdf_folder), candidate, f"{md_base}_pg{page}_repeat*.md")))
                    if md_files:
                        md_file_path = md_files[0]  # Use the first repeat as an example
                        with open(md_file_path, "r", encoding="utf-8") as f:
                            md_content = f.read()
                except Exception as e:
                    md_content = f"Error loading Markdown content: {str(e)}"

                # Add a button to toggle the Markdown content display
                if md_content:
                    html += f"""                    <button class="md-toggle" onclick="toggleMdContent('{candidate}_{pdf_name.replace('.', '_')}_{page}_md')">
                        Toggle Markdown Content
                    </button>
                    <div id="{candidate}_{pdf_name.replace('.', '_')}_{page}_md" class="md-content" style="display: none;">
{md_content}
                    </div>
"""

                tests = test_results_by_candidate[candidate][pdf_name][page]
                for test, passed, explanation in tests:
                    result_class = "pass" if passed else "fail"

                    # Start test div
                    html += f"""                <div class="test {result_class}">
                    <div class="test-header">
                        <div class="test-id">Test ID: {test.id}</div>
                        <div class="test-type">{test.type}</div>
                        <span class="status-badge {result_class}">{'Passed' if passed else 'Failed'}</span>
                    </div>
"""

                    # Add specific test details based on test type
                    test_type = getattr(test, "type", "").lower()

                    if test_type == "present" and hasattr(test, "text"):
                        text = getattr(test, "text", "")
                        html += f"""                    <div class="test-details"><strong>Text to find:</strong> "{text[:100]}{"..." if len(text) > 100 else ""}"</div>
"""
                    elif test_type == "absent" and hasattr(test, "text"):
                        text = getattr(test, "text", "")
                        html += f"""                    <div class="test-details"><strong>Text should not appear:</strong> "{text[:100]}{"..." if len(text) > 100 else ""}"</div>
"""
                    elif test_type == "order" and hasattr(test, "before") and hasattr(test, "after"):
                        before = getattr(test, "before", "")
                        after = getattr(test, "after", "")
                        html += f"""                    <div class="test-details"><strong>Text order:</strong> "{before[:50]}{"..." if len(before) > 50 else ""}" should appear before "{after[:50]}{"..." if len(after) > 50 else ""}"</div>
"""
                    elif test_type == "table":
                        if hasattr(test, "cell"):
                            cell = getattr(test, "cell", "")
                            html += f"""                    <div class="test-details"><strong>Table cell:</strong> "{cell}"</div>
"""
                        if hasattr(test, "up") and getattr(test, "up", None):
                            up = getattr(test, "up")
                            html += f"""                    <div class="test-details"><strong>Above:</strong> "{up}"</div>
"""
                        if hasattr(test, "down") and getattr(test, "down", None):
                            down = getattr(test, "down")
                            html += f"""                    <div class="test-details"><strong>Below:</strong> "{down}"</div>
"""
                        if hasattr(test, "left") and getattr(test, "left", None):
                            left = getattr(test, "left")
                            html += f"""                    <div class="test-details"><strong>Left:</strong> "{left}"</div>
"""
                        if hasattr(test, "right") and getattr(test, "right", None):
                            right = getattr(test, "right")
                            html += f"""                    <div class="test-details"><strong>Right:</strong> "{right}"</div>
"""
                    elif test_type == "math" and hasattr(test, "math"):
                        math = getattr(test, "math", "")
                        html += f"""                    <div class="test-details"><strong>Math equation:</strong></div>
                    <div class="math" data-math="{math.replace('"', '&quot;')}"></div>
"""

                    # Add explanation for failed tests
                    if not passed:
                        html += f"""                    <div class="explanation">Explanation: {explanation}</div>
"""

                    html += """                </div>
"""

                html += """            </div>
        </div>
    </div>
"""

        html += """    </div>
"""

    html += """    </div>
    
    <script>
        function switchTab(candidateId) {
            // Hide all content
            const contents = document.querySelectorAll('.candidate-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Deactivate all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Activate selected content and tab
            document.getElementById(candidateId).classList.add('active');
            document.querySelector(`.tab[onclick="switchTab('${candidateId}')"]`).classList.add('active');
        }
        
        function switchPageTab(baseId, pageNum) {
            // Hide all page content
            const pageContents = document.querySelectorAll(`[id^="${baseId}_"]`);
            pageContents.forEach(content => {
                if (content.classList.contains('page-content')) {
                    content.classList.remove('active');
                }
            });
            
            // Deactivate all page tabs
            const pageTabs = document.querySelectorAll(`#${baseId}_tabs .page-tab`);
            pageTabs.forEach(tab => tab.classList.remove('active'));
            
            // Activate selected page content and tab
            document.getElementById(`${baseId}_${pageNum}`).classList.add('active');
            document.querySelector(`#${baseId}_tabs .page-tab[onclick="switchPageTab('${baseId}', ${pageNum})"]`).classList.add('active');
        }
        
        function toggleMdContent(id) {
            const mdContent = document.getElementById(id);
            if (mdContent.style.display === "none") {
                mdContent.style.display = "block";
            } else {
                mdContent.style.display = "none";
            }
        }
        
        // Render math expressions
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false}
                ]
            });
            
            // Also render any elements with data-math attribute
            document.querySelectorAll('.math').forEach(function(el) {
                katex.render(el.getAttribute('data-math'), el, {
                    throwOnError: false,
                    displayMode: true
                });
            });
        });
    </script>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Enhanced HTML report generated: {output_file}")
