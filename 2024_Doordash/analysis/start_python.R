# # install pkg (in terminal)
# conda install pkg

# load pkg
library(reticulate)

# Point reticulate to the *conda* executable in /opt/anaconda3/bin
# which conda
use_condaenv("base", conda = "/opt/anaconda3/bin/conda", required = TRUE)
py_config()

reticulate::repl_python()

