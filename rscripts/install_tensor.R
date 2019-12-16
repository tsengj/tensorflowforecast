#https://github.com/rstudio/keras/issues/649
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")
tensorflow::install_tensorflow(method = "auto",version = "default") #"gpu" 'default'

library(keras)
install_keras(method = 'auto',
              conda = "auto", version = "default", tensorflow = "default") #"gpu" 'default'

# To activate this environment, use
# conda activate r-reticulate
# To deactivate an active environment, use
# conda deactivate

tensorflow::tf_config()

library(reticulate)
reticulate::py_config()

library(tensorflow)
rm -rf ~/.nv