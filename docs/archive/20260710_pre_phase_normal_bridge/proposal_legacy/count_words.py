#!/usr/bin/env python3
"""统计 PDF 文档词数"""

import subprocess
import re
import sys
from pathlib import Path

def count_pdf_words(pdf_path: str) -> dict:
    """提取 PDF 文本并统计词数"""
    result = subprocess.run(
        ['pdftotext', pdf_path, '-'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")

    text = result.stdout

    # 清理文本
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    words = clean_text.split()

    # 统计页数（通过 form feed 字符）
    pages = text.count('\f') + 1 if text else 0

    return {
        'total_words': len(words),
        'unique_words': len(set(w.lower() for w in words)),
        'pages': pages,
        'words_per_page': len(words) // pages if pages > 0 else 0,
        'characters': len(text),
    }

if __name__ == '__main__':
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else 'main.pdf'

    if not Path(pdf_file).exists():
        print(f"Error: {pdf_file} not found")
        sys.exit(1)

    stats = count_pdf_words(pdf_file)

    print(f"=== PDF Word Count: {pdf_file} ===")
    print(f"Total words:      {stats['total_words']:,}")
    print(f"Unique words:     {stats['unique_words']:,}")
    print(f"Pages:            {stats['pages']}")
    print(f"Words per page:   {stats['words_per_page']:,}")
    print(f"Characters:       {stats['characters']:,}")
