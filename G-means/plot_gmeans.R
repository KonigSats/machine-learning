df <- read.csv(file = "output.csv", header = TRUE)
data <- df[, -ncol(df)]
labels <- df[, ncol(df)]
plot(data, col = labels)