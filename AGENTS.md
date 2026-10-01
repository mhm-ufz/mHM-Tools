# General
- if a prompt is underspecified ask for clarification 
- ask before writing tests and show a plan for the test cases that I have to aprove of
- do not compile or run tests 
- Keep changes small and limited to the request.
- Write a docstring for every new function 
    - Docstrings should be short and concise. 
    - Document arguments (types if specified) and return arguments 
- use f-strings if working in python
- Keep code clear and concise. Do not create unnecessary functions. Also do not make it to short but keep it easily human readable. 
- When using comments to explain the following code, keep them concise and to three lines maximum.
- When I ask you to commit changes allways commit in small sections with meaningful commit messages and add the changes to the change log, ask for approval and then commit these as well in a seperate commit. 
- The changelog reflects the cumulative changes since the last tag, not just the current commit's diff. Before adding a new entry, review the existing unreleased entries and revise or merge any that this commit reworks or supersedes, rather than appending a separate entry for the same change.
- Commit messages are plain descriptive sentences (e.g. "Fix off-by-one bug in cut_to_filled_area's crop slices"), never conventional-commit prefixes like `fix:` or `feat:`. They  allways have active verbs in the beginning as if the sentence was: This commit will COMMIT_MESSAGE.
- Commit messages should be specific. e.g. when adding tests, state that you are adding tests. 
- When you use terminal commands and ask for permission write one maximum two lines of explenation. What is the goal of the command. Why do you need it. 
# Use Module:
- if writing a function look to `src/mhm-tools/common` to check if there are functions there that can be used. If their usage only slightly differs propose no breaking changes to the existing function. Do not implement it yourself. 
- all functions that only handle xarray DataArrays or DataSets put them in `src/mhm-tools/common/xarray_utils.md`


# Argument and Function Names
- allways use descriptive argument names. Use single letter arguments only for iterators in loops
- allways use descriptive function names. Ideally I can understand what the function does and returns from it name alone. 
Function names can for example start with:
    - `calculate`: caclulate a value from input
    - `create`: create an object or array or string from input
    - `get`: return a saved state from file or member variable (also from passed Object e.g. xarray dataset)
    - `set`: set value to passed argument
    - `write`: write to file
    - `read`: read from file 
    - `compare`: compare two or more passed arguments
- arguments discribing file or folder pathts should allways follow this logic: 
    - `_dir` discribes a directory path
    - `_file` discribes a file path
    - `_path` discribes a path that could either be a file or a directory. In this case there needs to be a point where it is checked what it is and is handled respectively. From then on `_dir` or `_file` name parts should be used again.
- CLI arguemtns should allways be dash seperated and python arguments by underscore
- Follow Python naming convention PEP 8

# Plot Creation
## General
- Use matplotlib. Put every plot in its own function, decorated with `@log_errors(raise_exceptions=True)`.
- Before plotting, replace `inf`/`-inf` with NaN. Mask ratios and percentages where the reference is close to zero.
- Reuse the existing helpers in `src/mhm_tools/common/plotter.py` (`plot_single_map`, `round_sensibly`) and don't reimplement bounds, ticks or rounding.
- Write the figure title with `fig.suptitle(..., fontweight="normal", fontsize="x-large")` in the form
  `"Comparison {input_name} with {ref_name} for years {first}-{last}"`. Leave out the years part if the period is unknown.
- When a figure has more than one panel, prefix each panel title with a letter (`a) `, `b) `, ...).
- Put the summary statistics in the panel title in parentheses with units, e.g. `(median=0.12mm/day, mean=0.10mm/day)`.
  Round them to the decimals the colorbar uses (`round_sensibly`), or `.2f` by default.
- Write units in square brackets on axis and colorbar labels, e.g. `ET [mm/day]`, `[%]`. Never leave a colorbar label empty.
  Take the variable name and units from the data instead of hard-coding them.

## Maps
- Draw gridded fields with `ax.imshow` through `plot_single_map`. Remove x/y ticks and use spine linewidth 0.25.
- Use discrete colorbars: `BoundaryNorm` with about 9 bins and the ticks at the bin edges (the dividers between colours).
  - Sequential maps: the outer edges are the colour limits (e.g. -0.41 for KGE), the inner edges sit on multiples of
    a nice step (e.g. -0.2, 0, 0.2, ...).
  - Diverging maps: an odd number of bins with one bin centred on the centre value (e.g. 0.9-1.1 around 1), so values
    close to it are easy to spot. Steps of 1, 2 or 5 times a power of ten; the limits are widened to the outer edges.
- Choose colormaps like this:
  - Use a diverging map (`coolwarm_r`) for ratios and differences. Centre it on the "no-difference" value
    (1 for ratios, 0 for differences) and keep the limits symmetric around that value.
  - Use a sequential map (`viridis_r`) for skill and correlation metrics. A correlation's lower limit follows the
    data, but drops to 0 or below as soon as values fall below it, so only negative correlations extend.
- Choose the colour limits separately for each end of the colour scale:
  - By default, use the data min/max.
  - Use the percentile (1st/99th, or 5th/95th) instead when outliers stretch the range, i.e. when
    `|max - p99| > |p99 - mean|` (and likewise `|p1 - min| > |mean - p1|` for the lower end).
  - For diverging maps, then use the larger distance from the centre on both sides.
  - Round the limits outwards, never onto the data: sequential limits to multiples of the bin step, diverging limits
    to the outer edge of the centred bin layout.
  - Set `extend` on the colorbar whenever data fall outside the bounds.
- For a single map, attach the colorbar on the right with `make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)`.
  For small multiples (e.g. 12 monthly maps), use the same limits on every panel and one shared colorbar in its own gridspec column.

## Line / bar plots
- Use the same colours for the same roles in every plot: reference `#008176`, input/model `#79A3E6` (bars with `alpha=0.8`),
  ratio input/reference `#0000A7` (dashed line).
- Draw input and reference as side-by-side bars of width 0.4.
- Plot ratios on a `twinx` axis with a thin (`linewidth=0.5`) horizontal line at 1. Colour the twin axis label and ticks like the ratio line.
- Combine the handles from both axes into a single legend (`loc="upper right"`).
- Label the months 1–12 on the x axis as `month of year`. Use spine linewidth 0.5 on non-map axes.

## Layout and output
- Use a figure width of 10.5 in for standard figures. Wider is fine for large grids of small multiples (e.g. 14 x 8.5).
  Use `fig.add_gridspec` for uneven layouts, e.g. to centre a lower panel.
- Call `plt.tight_layout()` (or `fig.subplots_adjust` for gridspec small multiples). Save as PNG with dpi 400.
- Name files `{plot_kind}_{input_name}_{ref_name}.png` and replace spaces with `_`. Build the path with `pathlib` (`output_dir / file_name`).
- Always `plt.close(fig)` after saving, and log the written path with `logger.info`.
