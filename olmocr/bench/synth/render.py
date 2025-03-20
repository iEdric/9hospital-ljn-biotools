#!/usr/bin/env python3
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright

# Simple configuration
CONFIG = {
    "input_file": os.path.join(os.path.dirname(__file__), "templates", "listpage.js"),  # React component file
    "output_pdf": "book-page.pdf",  # Output PDF filename
    "temp_html": "temp-render.html",  # Temporary HTML file
    "wait_time": 1500,  # Time to wait for rendering (ms)
    "device_scale": 2,  # Resolution multiplier
    "debug": True,  # Keep temp files for debugging
}


async def create_html_file():
    """Create a temporary HTML file that loads the React component from a file."""
    try:
        # Check if input file exists
        input_path = Path(CONFIG["input_file"])
        if not input_path.exists():
            print(f"Error: Input file '{input_path}' not found")
            return False

        # Read the component file
        with open(input_path, "r", encoding="utf-8") as f:
            component_code = f.read()

        # Create HTML that will load our component
        html_content = (
            """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Book Page Template</title>
  <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    * {
      box-sizing: border-box;
    }
    
    html, body {
      margin: 0;
      padding: 0;
      width: 8.5in;
      height: 11in;
      overflow: hidden;
    }
    
    #root {
      width: 100%;
      height: 100%;
      padding: 0.25in;
      overflow: hidden;
    }
    
    @media print {
      body {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    // The React component code loaded from external file
    """
            + component_code
            + """
    
    // Render only the book page part, not the controls
    ReactDOM.render(
      <BookPageTemplate />,
      document.getElementById('root')
    );
  </script>
</body>
</html>
        """
        )

        with open(CONFIG["temp_html"], "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Created HTML file: {CONFIG['temp_html']}")
        print(f"Using React component from: {CONFIG['input_file']}")
        return True
    except Exception as e:
        print(f"Error creating HTML file: {e}")
        print(f"Exception details: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def render_to_pdf():
    """Render the React component to PDF using Playwright."""
    try:
        # Create the HTML file first
        html_created = await create_html_file()
        if not html_created:
            print("Failed to create HTML file")
            return

        print("Launching browser...")
        async with async_playwright() as p:
            # Launch the browser with more debugging options
            browser = await p.chromium.launch(
                headless=True,  # True for production, False for debugging
            )

            # Create a new page for letter size paper
            page = await browser.new_page(viewport={"width": 816, "height": 1056}, device_scale_factor=CONFIG["device_scale"])  # 8.5in x 11in at 96dpi

            # Get absolute path to HTML file
            html_path = Path(CONFIG["temp_html"]).absolute()
            html_uri = f"file://{html_path}"

            print(f"Navigating to: {html_uri}")

            # Add event listeners for console messages and errors
            page.on("console", lambda msg: print(f"Browser console: {msg.text}"))
            page.on("pageerror", lambda err: print(f"Browser page error: {err}"))

            # Navigate with longer timeout and wait for network idle
            await page.goto(html_uri, wait_until="networkidle", timeout=30000)

            # Wait for React to render
            await page.wait_for_timeout(CONFIG["wait_time"])

            # Add a check to ensure the component rendered
            element_count = await page.evaluate(
                """() => {
                const root = document.getElementById('root');
                return root.childElementCount;
            }"""
            )

            if element_count == 0:
                print("Warning: No elements found in root. Component may not have rendered.")
            else:
                print(f"Found {element_count} elements in root. Component rendered successfully.")

            # Save debug screenshot
            if CONFIG["debug"]:
                await page.screenshot(path="debug-screenshot.png")
                print("Debug screenshot saved")

            # Generate PDF
            print("Generating PDF...")
            await page.pdf(path=CONFIG["output_pdf"], format="Letter", print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})

            print(f"PDF generated successfully: {CONFIG['output_pdf']}")

            # Close the browser
            await browser.close()

        # Cleanup temp files if not in debug mode
        if not CONFIG["debug"] and Path(CONFIG["temp_html"]).exists():
            Path(CONFIG["temp_html"]).unlink()
            print("Temporary HTML file removed")

    except Exception as e:
        print(f"Error generating PDF: {e}")


if __name__ == "__main__":
    # Run the async function
    try:
        asyncio.run(render_to_pdf())
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
