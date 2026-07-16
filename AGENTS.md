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
