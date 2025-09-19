# Building documents

## Installing dependencies
 - Via `pip` (from repo's root directory)
     ```
        pip install . [docs]    
     ```
 - Using `conda` environment
    ```
       conda create --file ./devtools/conda-envs/docs_env.yaml
    ```
## Create `.rst` files for modules
This is only needed if the files are not present for some modules.
```
    sphinx-apidoc -o docs/source/ src/neuripp/
```
## Build documents
This has to be done when the code changes
```
    cd docs
    make html  
```
(for other possible output formats use `make help`).
Compiled documentation will appear in `docs/build/html`.

## Adding examples
`.ipynb` files added to `examples/notebooks` can be automatically converted to examples in the documentation.
One has to append to `docs/source/_include/examples.rst` the following line:
```
   The title of the new example <notbooks/NEW_EXAMPLE_NAME.ipynb> 
```
