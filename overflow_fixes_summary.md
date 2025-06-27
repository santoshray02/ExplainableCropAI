# Overflow Fixes Summary - ExplainableCropAI Paper

## Issues Fixed

### 1. **Table Overflow - Trust and Workload Table**
- **Problem**: Table with 6 columns was too wide for IEEE two-column format
- **Solution**: Added `\resizebox{\columnwidth}{!}{...}` to scale table to fit column width
- **File**: `paper/results_discussion.tex`

### 2. **Python Code Overflow**
- **Problem**: Long lines in Python code listings were extending beyond column width
- **Solution**: 
  - Added `basicstyle=\footnotesize\ttfamily` to reduce font size
  - Added `breaklines=true` to enable automatic line breaking
  - Manually broke long lines at appropriate points
- **Files**: `paper/dataset_implementation.tex`
- **Code blocks fixed**:
  - PaddyDiseaseClassifier class definition
  - estimate_uncertainty function

### 3. **Figure Overflow**
- **Problem**: All figures were using `width=0.9\textwidth` which is too wide for two-column format
- **Solution**: Changed all figure widths from `0.9\textwidth` to `\columnwidth`
- **Files fixed**:
  - `paper/figures.tex` - 8 figures updated
  - `paper/results_discussion.tex` - 2 figures updated
  - Added missing user interface figure definition

## Technical Details

### IEEE Two-Column Format
- `\textwidth` = full page width (both columns)
- `\columnwidth` = single column width
- For IEEE conference format, figures should use `\columnwidth` not `\textwidth`

### LaTeX Commands Used
1. **For tables**: `\resizebox{\columnwidth}{!}{...}`
   - Scales table to fit column width while maintaining aspect ratio
   
2. **For code**: 
   ```latex
   \begin{lstlisting}[language=Python, basicstyle=\footnotesize\ttfamily, breaklines=true]
   ```
   - `\footnotesize` reduces font size
   - `\ttfamily` uses monospace font
   - `breaklines=true` enables automatic line breaking

3. **For figures**: `\includegraphics[width=\columnwidth]{...}`
   - Ensures figures fit within single column

## Files Modified

1. **paper/results_discussion.tex**
   - Fixed trust-workload table with resizebox
   - Fixed 2 figure references to use columnwidth
   - Fixed typo in figure closing tag

2. **paper/dataset_implementation.tex**
   - Fixed 2 Python code listings with proper formatting

3. **paper/figures.tex**
   - Updated all 8 figures to use columnwidth
   - Added user interface figure definition

## Verification

To verify these fixes work correctly:

1. Compile the LaTeX document
2. Check that no content extends beyond column boundaries
3. Verify tables are readable (not too small)
4. Ensure code is properly formatted with line breaks
5. Confirm all figures fit within columns

## Additional Recommendations

If any content still overflows after these fixes:

1. **For tables**: Consider splitting wide tables into two separate tables
2. **For code**: Further reduce font size or abbreviate variable names
3. **For figures**: Use subfigures or create multi-panel figures
4. **General**: Use landscape orientation for very wide content (with `\usepackage{rotating}` and `\begin{sidewaystable}`)

All overflow issues should now be resolved for proper IEEE two-column formatting.
