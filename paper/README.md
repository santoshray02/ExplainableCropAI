# ExplainableCropAI - Springer Nature (SN) Article Format

This folder contains the ExplainableCropAI paper converted to Springer Nature article template format.

## Contents

- `main.tex` - Main LaTeX document in SN format
- `references.bib` - Bibliography file
- `sn-jnl.cls` - Springer Nature journal class file
- `bst/` - Bibliography style files directory
- `paper/figures/` - All figures in PDF format

## Key Changes from MDPI Format

1. **Document Class**: Changed from `\documentclass[agriculture,article,submit,pdftex,moreauthors]{Definitions/mdpi}` to `\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}`

2. **Author Format**: Converted from MDPI's format to SN's structured format:
   - Uses `\fnm{}` for first name and `\sur{}` for surname
   - Affiliations use `\affil[]{}` with structured fields (`\orgdiv`, `\orgname`, `\orgaddress`, etc.)

3. **Abstract**: Simplified to unstructured format (SN template supports both structured and unstructured)

4. **Keywords**: Changed from `\keyword{}` to `\keywords{}`

5. **Section Labels**: Added descriptive labels to all sections for better cross-referencing

6. **Tables**: Updated to use SN's preferred format with `\toprule`, `\midrule`, and `\botrule` from booktabs

7. **Declarations Section**: Added comprehensive declarations section at the end with:
   - Funding
   - Conflict of interest
   - Ethics approval
   - Consent to participate
   - Consent for publication
   - Data availability
   - Code availability
   - Author contribution
   - Acknowledgements

8. **Bibliography**: Uses `\bibliography{references}` instead of MDPI's `\externalbibliography` approach

9. **Figure Paths**: Updated figure paths to use relative paths (`../paper/figures/`)

## Compilation Instructions

**IMPORTANT**: All `.bst` bibliography style files have been copied to the main directory to ensure BibTeX can find them. This is required for the references to compile correctly.

To compile the document:

1. **Using pdfLaTeX** (recommended):
   ```bash
   cd c:\Users\raysk\projects\mtech\ExplainableCropAI\sn-format
   pdflatex main.tex    # First pass - generates .aux file
   bibtex main          # Process bibliography (requires .bst files in current dir)
   pdflatex main.tex    # Second pass - include bibliography
   pdflatex main.tex    # Third pass - resolve all references
   ```

2. **Using LaTeX Workshop in VS Code**:
   - Open `main.tex`
   - Use the "Build LaTeX project" command (Ctrl+Alt+B on Windows/Linux)

3. **Using Overleaf**:
   - Upload all files to Overleaf
   - The document should compile automatically
   - Make sure to upload the entire folder structure including `bst/` and `paper/figures/`

## Reference Style

The document uses `sn-mathphys-num` style which is:
- **Math and Physical Sciences Numbered Reference Style**
- Numbered citations in the text
- Suitable for computer science and engineering journals

If you need a different reference style, change the document class option:
- `sn-basic` - Basic Springer Nature Reference Style
- `sn-nature` - Nature Portfolio journals
- `sn-mathphys-ay` - Math and Physical Sciences Author Year
- `sn-aps` - American Physical Society
- `sn-vancouver-num` - Vancouver Numbered
- `sn-chicago` - Chicago-based Humanities

## Notes

- All content from the original MDPI format has been preserved
- Mathematical equations, algorithms, and code listings are compatible with the SN template
- Figure and table formatting follows SN guidelines
- The paper structure (Introduction, Materials and Methods, Results, Discussion, Conclusion) is maintained

## Troubleshooting

If you encounter compilation errors:

1. **Missing figures**: Ensure the `paper/figures/` directory contains all PDF figures
2. **Bibliography errors**: Run `bibtex` after the first `pdflatex` compilation
3. **Class file errors**: Ensure `sn-jnl.cls` and the `bst/` directory are in the same folder as `main.tex`
4. **Package conflicts**: The template includes all necessary packages; avoid adding conflicting packages

## Contact

For questions about the paper content, contact:
- **Corresponding Author**: Sumit Kumar (sumitkumar@sitpune.edu.in)

For template-related issues, refer to the [Springer Nature LaTeX template documentation](https://www.springernature.com/gp/authors/campaigns/latex-author-support).
