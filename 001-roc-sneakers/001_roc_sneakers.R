library(tidyverse)
library(e1071)
library(caret)
library(pROC)
library(ggplot2)

# Simulate fake Nike data ----
set.seed(42)

n_humans <- 600
n_bots   <- 400

# 1. Humans: Now some are faster and use newer accounts
humans <- tibble(
  is_bot = 'Human',
  click_speed_ms = rnorm(n_humans, mean = 650, sd = 300),    # Frantic humans are fast
  account_age_days = rnorm(n_humans, mean = 100, sd = 60),   # Newer fans
  ip_risk_score = runif(n_humans, 20, 70),                   # VPNs and public Wi-Fi
  session_duration_s = rnorm(n_humans, mean = 80, sd = 40),  # In and out quickly
  request_variance = rnorm(n_humans, mean = 250, sd = 120),  # Some rhythm to their clicking
  device_id_count = sample(1:4, n_humans, replace = TRUE, prob = c(0.7, 0.2, 0.07, 0.03))
)

# 2. Bots: Now they try to act 'Human' (Slower, older proxies)
bots <- tibble(
  is_bot = 'Bot',
  click_speed_ms = rnorm(n_bots, mean = 550, sd = 300),     # Throttled bots mimic human speed
  account_age_days = rnorm(n_bots, mean = 80, sd = 60),      # 'Aged' bot accounts
  ip_risk_score = runif(n_bots, 35, 85),                     # High-quality residential proxies
  session_duration_s = rnorm(n_bots, mean = 65, sd = 40),    # Simulated browsing
  request_variance = rnorm(n_bots, mean = 200, sd = 120),    # Artificial 'jitter' added to scripts
  device_id_count = sample(2:6, n_bots, replace = TRUE)      # Spreading accounts across devices
)

# 3. Combine and Shuffle
snkrs_data <- bind_rows(humans, bots) %>%
  mutate(is_bot = factor(is_bot, levels = c('Human', 'Bot'))) %>%
  slice_sample(prop = 1)                                  # Shuffle the entries




# Fit a model ----


# 1. Split into Training (80%) and Testing (20%)
set.seed(123)
train_index <- createDataPartition(snkrs_data$is_bot, p = 0.8, list = FALSE)
train_set   <- snkrs_data[train_index, ]
test_set    <- snkrs_data[-train_index, ]

# 2. Train the Naive Bayes Classifier
# We are predicting 'is_bot' using all other columns (.)
nb_model <- naiveBayes(is_bot ~ ., data = train_set)

# 3. Predict Probabilities on the Test Set
# We need 'type = raw' to get the 0.0 to 1.0 probability scores for the ROC curve
probs <- predict(nb_model, test_set, type = 'raw')
test_results <- test_set %>%
  mutate(prob_bot = probs[, 'Bot'])


# Plot the curve ----

# 1. Generate the ROC Curve Data
snkrs_roc <- roc(test_results$is_bot,
                 test_results$prob_bot,
                 levels = c('Human', 'Bot'))


# 2. Extract coordinates into a dataframe for ggplot
roc_df <- data.frame(
  tpr = snkrs_roc$sensitivities,
  fpr = 1 - snkrs_roc$specificities
) %>%
  arrange(fpr, tpr)

# 3. Create the ggplot ROC curve
ggplot(roc_df, aes(x = fpr, y = tpr)) +
  # The ROC Line (Nike Heritage Red)
  geom_line(color = '#E31837', size = 1) +
  # The 'Random Guess' reference line
  geom_abline(linetype = 'dashed', color = '#333333', alpha = 0.5) +
  # Labels and Annotations
  labs(
    title = 'ROC Curve: SNKRS Bot Detection',
    x = 'False Positive Rate\n(1 - Specificity)',
    y = 'True Positive Rate\n(Sensitivity)'
  ) +
  # Styling for a clean, modern 'Data Snack' look
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = 'bold', size = 18),
    plot.subtitle = element_text(color = 'grey40', margin = margin(b = 15)),
    panel.grid.minor = element_blank(),
    axis.title.y = element_text(angle = 0, vjust = 0.5, margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10))
  ) +
  # Ensure the plot is a perfect square
  coord_fixed()

ggsave('data_snack_001.png', dpi=300, width=10, height=8)
