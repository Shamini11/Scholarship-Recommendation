Scholarship recommendation research paper LaTeX package

Files:
- main.tex: IEEE conference-style LaTeX source.
- figures/: all figure images referenced by main.tex.

How to use in Overleaf:
1. Create a new blank project.
2. Upload main.tex and the entire figures folder (or upload the ZIP package).
3. Set main.tex as the main document if necessary.
4. Compile with pdfLaTeX.
5. Replace Author Name, Institution Name, and email placeholders.

IMPORTANT DATA VALIDATION NOTES:
1. The supplied scholarship-type chart shown earlier with 8 categories (18, 12, 8, 6, 5, 4, 3, 4) does NOT match the current workbook's Scholarship_Type column, which contains 23 literal categories. To avoid reporting unsupported counts, the package uses a graph generated from the actual workbook instead.
2. The supplied case-study evaluation chart contains similarity scores and eligibility counts. Verify that these exact values came from a reproducible run of the prototype before presenting them as empirical results. The LaTeX caption flags this requirement.
3. The paper does not claim accuracy/precision/recall/F1 because the current dataset has no labelled student-scholarship outcome pairs.
4. Architecture and workflow figures are conceptual/proposed-process diagrams; they do not establish that every module is implemented.
