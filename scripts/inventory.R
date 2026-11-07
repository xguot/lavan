# Parse the reference AST rather than estimating function counts with text regexes.
source("tests/r/reference.R")
records <- list()
visit <- function(expr, file, line) {
  if (!is.call(expr)) return(invisible(NULL))
  head <- as.character(expr[[1]])[[1]]
  if (head %in% c("<-", "=")) {
    if (length(expr) != 3L || !is.symbol(expr[[2]])) return(invisible(NULL))
    name <- as.character(expr[[2]])
    rhs <- expr[[3]]
    kind <- if (is.call(rhs)) as.character(rhs[[1]])[[1]] else ""
    if (kind %in% c("function", "lav_alias")) {
      target <- if (kind == "lav_alias") as.character(rhs[[2]]) else NULL
      records[[length(records) + 1L]] <<- list(name = name, file = file, line = line,
                                             kind = if (kind == "function") "function" else "alias", target = target)
    }
    return(invisible(NULL))
  }
  if (head == "function") return(invisible(NULL))
  for (child in as.list(expr)[-1]) visit(child, file, line)
}
files <- sort(list.files(file.path(source_dir, "R"), pattern = "\\.R$", full.names = TRUE))
for (file in files) {
  parsed <- parse(file, keep.source = TRUE)
  refs <- attr(parsed, "srcref")
  for (i in seq_along(parsed)) visit(parsed[[i]], paste0("R/", basename(file)), as.integer(refs[[i]][1]))
}
emit(list(version = expected_version, r_files = length(files),
          r_lines = sum(vapply(files, function(f) length(readLines(f, warn = FALSE)), integer(1))),
          exports = sort(getNamespaceExports("lavaan")), definitions = records))
