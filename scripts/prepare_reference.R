#!/usr/bin/env Rscript
# Install exactly the local reference and stamp every relevant source file.
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(if (length(args)) args[[1]] else ".")
source_dir <- file.path(root, "reference", "lavaan")
lib <- file.path(root, ".tools", "R")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
status <- system2(file.path(R.home("bin"), "R"), c("CMD", "INSTALL", paste0("--library=", shQuote(lib)), shQuote(source_dir)))
if (status != 0L) stop("reference installation failed")
relative <- c("DESCRIPTION", "NAMESPACE",
              file.path("R", sort(list.files(file.path(source_dir, "R"), pattern = "\\.R$"))),
              file.path("data", sort(list.files(file.path(source_dir, "data")))))
hash <- tools::md5sum(file.path(source_dir, relative))
names(hash) <- relative
saveRDS(hash, file.path(lib, "lavaan", "lavan-source-md5.rds"))
cat("Installed and stamped ", length(hash), " reference files\n", sep = "")
