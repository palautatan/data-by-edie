library(shiny)
library(DBI)
library(duckdb)
library(ggplot2)
library(dplyr)
library(visNetwork)

# ==========================================
# 1. SETUP: Read from DuckDB & CSV
# ==========================================
con <- dbConnect(duckdb::duckdb(), "data/love_island_reddit.duckdb", read_only = TRUE)

# Survival & Prediction Data
cox_data <- dbGetQuery(con, "SELECT * FROM dashboard.cox_survival_probabilities")
logistic_data <- dbGetQuery(con, "SELECT * FROM dashboard.logistic_episode_probabilities")

# Network & Kalshi Data
overall_nodes <- dbGetQuery(con, "
  SELECT target_islander AS id, target_islander AS label, 
         num_comments AS value, community AS group 
  FROM dashboard.overall_network_nodes
")
overall_edges <- dbGetQuery(con, "
  SELECT source AS 'from', target AS 'to', weight AS value 
  FROM dashboard.overall_network_edges
")

episode_nodes <- dbGetQuery(con, "
  SELECT target_islander AS id, target_islander AS label, 
         num_comments AS value, community AS group, 
         episode, kalshi_price 
  FROM dashboard.episode_network_nodes
")
episode_edges <- dbGetQuery(con, "
  SELECT source AS 'from', target AS 'to', weight AS value, episode 
  FROM dashboard.episode_network_edges
")

dbDisconnect(con, shutdown = TRUE)

# Load lifecycle metadata to track active villa presence
lifecycle <- read.csv("data/metadata/islander_metadata.csv")

# Extract unique choices for dropdown menus
islander_names <- sort(unique(cox_data$target_islander))
episodes <- sort(unique(episode_nodes$episode))
episode_choices <- c("Overall" = "overall", setNames(episodes, paste("Episode", episodes)))


# ==========================================
# 2. UI: Two-Tab Layout
# ==========================================
ui <- navbarPage(
  title = "Love Island Insights",
  
  # TAB 1: Network (Unchanged)
  tabPanel("Network",
           sidebarLayout(
             sidebarPanel(
               selectInput("network_view", "Choose View:", choices = episode_choices),
               hr(),
               helpText("Network Graph Controls: Scroll to zoom, drag to pan, and click a node to highlight their connections.")
             ),
             mainPanel(
               visNetworkOutput("networkPlot", height = "650px")
             )
           )
  ),
  
  # TAB 2: Combined Contestant Metrics
  tabPanel("Contestant Metrics",
           sidebarLayout(
             sidebarPanel(
               selectInput("metrics_person", "Choose an Islander:", choices = islander_names),
               hr(),
               helpText("These charts strictly display data for the episodes where the contestant was actively present in the villa.")
             ),
             mainPanel(
               # Stack the three plots vertically
               plotOutput("survivalPlot", height = "280px"),
               plotOutput("logisticPlot", height = "280px"),
               plotOutput("kalshiPlot", height = "280px")
             )
           )
  )
)


# ==========================================
# 3. SERVER: Reactive Logic
# ==========================================
server <- function(input, output, session) {
  
  # --- Network Logic ---
  output$networkPlot <- renderVisNetwork({
    if (input$network_view == "overall") {
      n_data <- overall_nodes
      e_data <- overall_edges %>% filter(value >= quantile(value, 0.50))
    } else {
      ep_num <- as.numeric(input$network_view)
      active_islanders <- lifecycle %>% 
        filter(entrance_episode <= ep_num & exit_episode >= ep_num) %>% 
        pull(islander)
      
      n_data <- episode_nodes %>% 
        filter(episode == ep_num, id %in% active_islanders)
      e_data <- episode_edges %>% 
        filter(episode == ep_num, from %in% active_islanders, to %in% active_islanders) %>% 
        filter(value >= quantile(value, 0.50, na.rm = TRUE))
    }
    
    visNetwork(n_data, e_data) %>%
      visOptions(highlightNearest = TRUE) %>% 
      visPhysics(solver = "forceAtlas2Based", stabilization = TRUE) %>%
      visEdges(color = list(opacity = 0.5))
  })
  
  # --- METRICS: Survival Curve (Cox) ---
  output$survivalPlot <- renderPlot({
    life <- lifecycle %>% filter(islander == input$metrics_person)
    
    # Filter to active followup
    plot_data <- cox_data %>% 
      filter(target_islander == input$metrics_person, within_observed_followup == TRUE)
    
    # Changed x from tenure_episode to calendar_episode
    ggplot(plot_data, aes(x = calendar_episode, y = survival_probability)) +
      geom_step(color = "#287268", linewidth = 1.2) +
      scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
      scale_x_continuous(limits = c(life$entrance_episode, life$exit_episode)) + # Force axis alignment
      labs(
        title = "Cox Baseline Model: Overall Survival Probability", 
        x = "Broadcast Episode", 
        y = "Survival %"
      ) +
      theme_minimal(base_size = 13)
  })
  
  # --- METRICS: Logistic Exit Risk ---
  output$logisticPlot <- renderPlot({
    life <- lifecycle %>% filter(islander == input$metrics_person)
    
    plot_data <- logistic_data %>% 
      filter(target_islander == input$metrics_person, 
             prediction_status == "predicted",
             target_episode >= life$entrance_episode,
             target_episode <= life$exit_episode)
    
    ggplot(plot_data, aes(x = target_episode, y = booted_next_episode_probability)) +
      geom_line(color = "#b0413e", linewidth = 1.2) +
      geom_point(color = "#b0413e", size = 2.5) +
      scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
      scale_x_continuous(limits = c(life$entrance_episode, life$exit_episode)) + # Force axis alignment
      labs(
        title = "Logistic Model: Next-Episode Exit Risk", 
        x = "Broadcast Episode", 
        y = "Exit Risk %"
      ) +
      theme_minimal(base_size = 13)
  })
  
  # --- METRICS: Kalshi Price ---
  output$kalshiPlot <- renderPlot({
    life <- lifecycle %>% filter(islander == input$metrics_person)
    
    plot_data <- episode_nodes %>% 
      filter(id == input$metrics_person,
             episode >= life$entrance_episode,
             episode <= life$exit_episode)
    
    ggplot(plot_data, aes(x = episode, y = kalshi_price)) +
      geom_line(color = "#d18a66", linewidth = 1.2) +
      geom_point(color = "#d18a66", size = 2.5) +
      scale_x_continuous(limits = c(life$entrance_episode, life$exit_episode)) + # Force axis alignment
      labs(
        title = "Prediction Market: Kalshi Win Probability", 
        x = "Broadcast Episode", 
        y = "Price (¢)"
      ) +
      theme_minimal(base_size = 13)
  })
}

shinyApp(ui = ui, server = server)