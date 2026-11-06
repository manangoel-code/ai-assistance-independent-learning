# Optional R counterpart of the primary models in analysis.py.
# Reproduces the four primary treatment-versus-control estimates, classroom-clustered
# standard errors, t intervals with G - 1 degrees of freedom and Holm-adjusted p values.
# Agrees with results/regression_results.csv to about 1e-14 (R 4.6.1, sandwich).
#
# Setup:  install.packages("sandwich")
# Run from the repository root after download_data.py:  Rscript analysis.R
library(sandwich)

data <- read.csv("research_data/main_regressions/final_data.csv")
data <- data[data$Honors == 0, ]
needed <- c("Part2Tot", "Part3Tot", "GPTBase", "GPTTutor", "gpa_prev",
            "teacher", "Session", "Grader", "Year", "Class")
data <- data[complete.cases(data[needed]), ]
data$teacher <- factor(data$teacher)
data$Session <- factor(data$Session)
data$Grader <- factor(data$Grader)
data$Year <- factor(data$Year)

practice <- lm(Part2Tot ~ GPTBase + GPTTutor + gpa_prev +
                 teacher + Session + Grader + Year, data = data)
exam <- lm(Part3Tot ~ GPTBase + GPTTutor + gpa_prev +
             teacher + Session + Grader + Year, data = data)

get_results <- function(model, outcome) {
  covariance <- vcovCL(model, cluster = data$Class, type = "HC1", cadjust = TRUE)
  names <- c("GPTBase", "GPTTutor")
  estimate <- coef(model)[names]
  se <- sqrt(diag(covariance))[names]
  df <- length(unique(data$Class)) - 1
  data.frame(outcome = outcome, condition = names, estimate = estimate,
             se = se, lower = estimate - qt(.975, df) * se,
             upper = estimate + qt(.975, df) * se,
             p = 2 * pt(-abs(estimate / se), df))
}

results <- rbind(get_results(practice, "Practice"), get_results(exam, "Exam"))
results$holm_p <- p.adjust(results$p, method = "holm")
print(results)
dir.create("results", showWarnings = FALSE)
write.csv(results, "results/r_primary_results.csv", row.names = FALSE)
