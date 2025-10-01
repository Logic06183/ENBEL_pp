# Install required packages for DLNM analysis
packages <- c("dlnm", "mgcv", "splines", "ggplot2", "dplyr", "tidyr", "jsonlite", "corrplot", "viridis", "gridExtra")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing package:", pkg, "\n")
    install.packages(pkg, repos = "https://cran.r-project.org")
    library(pkg, character.only = TRUE)
  } else {
    cat("Package", pkg, "already installed\n")
  }
}

cat("All packages installed successfully!\n")