library("pacman")
p_load("arrow")

load_eval_dataset <- function() {
  path <- "E:/shared_resources/t2d/model_eval/chapelet-megaloblastic/rural-silence-2005/evaluation_dataset.parquet"
  dataset <- read_parquet(path)
  
  
  return(dataset)
}

load_synth_eval_dataset <- function() {
  path <- "E:/shared_resources/t2d/model_eval/chapelet-megaloblastic/rural-silence-2005/evaluation_dataset.parquet"
  dataset <- read_parquet(path)
  
  
  return(dataset)
}