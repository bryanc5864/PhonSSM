# PhonSSM ICML 2026 Paper

Paper: **PhonSSM: Phonology-Aware State Space Model for Sign Language Recognition**

## Quick Start

1. **Generate figures**:
   ```bash
   python scripts/create_figures.py
   ```

2. **Compile the paper**:
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

   Or using the Makefile:
   ```bash
   make
   ```

## Files

| File | Description |
|------|-------------|
| `main.tex` | Main LaTeX document |
| `references.bib` | BibTeX references |
| `icml2026.sty` | ICML style file |
| `icml2026.bst` | ICML bibliography style |
| `fancyhdr.sty` | Header/footer package |
| `figures/` | Generated figures |
| `scripts/` | Figure generation scripts |

## Paper Structure

1. **Abstract** - Qualitative summary of contributions
2. **Introduction** - Motivation and key insight
3. **Background** - Sign language phonology primer
4. **Method** - AGAN, PDM, BiSSM, HPC components
5. **Experiments** - WLASL and Merged-5565 results
6. **Related Work** - Prior approaches
7. **Discussion** - Findings and limitations
8. **Appendix** - Extended methods and results

## Key Results

- WLASL100: 88.37% (skeleton-only SOTA)
- WLASL300: 74.41%
- WLASL1000: 62.90%
- WLASL2000: 72.08%
- Merged-5565: 53.34% (vs 27.39% baseline)

## Requirements

### LaTeX
- pdflatex
- bibtex
- Standard packages (amsmath, graphicx, booktabs, etc.)

### Python (for figures)
```bash
pip install numpy matplotlib
```

## Citation

```bibtex
@inproceedings{zhang2026phonssm,
  title={PhonSSM: Phonology-Aware State Space Model for Sign Language Recognition},
  author={Zhang, Jasper and Cheng, Bryan},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
