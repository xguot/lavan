# Refuse global lavaan and stale local installations; never silently skip.
root <- normalizePath(Sys.getenv("LAVAN_ROOT", "."))
lib <- file.path(root, ".tools", "R")
source_dir <- file.path(root, "reference", "lavaan")
stamp <- file.path(lib, "lavaan", "lavan-source-md5.rds")
if (!file.exists(stamp)) stop("Run Rscript scripts/prepare_reference.R before equivalence tests")
expected <- readRDS(stamp)
actual <- tools::md5sum(file.path(source_dir, names(expected)))
if (!identical(unname(actual), unname(expected))) stop("Reference changed: reinstall with scripts/prepare_reference.R")
source_files <- c("DESCRIPTION", "NAMESPACE",
                  file.path("R", sort(list.files(file.path(source_dir, "R"), pattern = "\\.R$"))),
                  file.path("data", sort(list.files(file.path(source_dir, "data")))))
if (!identical(names(expected), source_files)) stop("Reference source-file set changed: reinstall")
suppressPackageStartupMessages(library(lavaan, lib.loc = lib))
if (normalizePath(find.package("lavaan")) != normalizePath(file.path(lib, "lavaan"))) stop("wrong lavaan namespace")
expected_version <- read.dcf(file.path(source_dir, "DESCRIPTION"))[1, "Version"]
if (packageDescription("lavaan")$Version != expected_version) stop("wrong lavaan version")
lav <- function(.name, ...) get(.name, envir = asNamespace("lavaan"))(...)
mat <- function(a) list(rows = nrow(a), cols = ncol(a), data = as.vector(a))
emit <- function(x) cat(jsonlite::toJSON(x, digits = 17, auto_unbox = TRUE, na = "string", null = "null"))
