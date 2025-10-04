from docling.document_converter import DocumentConverter
from pathlib import Path

def convert_to_markdown(file_paths):
    converter = DocumentConverter()

    for file_path in file_paths:
        input_path = Path(file_path)
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            continue

        # Parse the file
        print(f"📄 Parsing: {input_path.name}")
        result = converter.convert(input_path)

        # Create output Markdown path (same directory)
        output_path = input_path.with_suffix(".md")

        # Save the parsed document to Markdown
        markdown_content = result.document.export_to_markdown()
        output_path.write_text(markdown_content, encoding="utf-8")

        print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    # Example: three files in the same directory
    files_to_convert = [
        "/home/david/hackathon_oct_2025/data/annual_reports/Erste_Group_2024.pdf",
        "/home/david/hackathon_oct_2025/data/annual_reports/GSK_esg-performance-report_2023.pdf",
        "/home/david/hackathon_oct_2025/data/annual_reports/swisscom_sustainability_impact_report_2024_en.pdf"
    ]

    convert_to_markdown(files_to_convert)
