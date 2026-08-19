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
