# Copyright 2023-2024 ETH Zurich. All rights reserved.


import pypandoc

pypandoc.download_pandoc()

output = pypandoc.convert_file(
    "theoretical_analysis.md", "pdf", outputfile="theoretical_analysis.pdf"
)

assert output == ""
