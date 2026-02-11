library(tidyverse)
library(ggridges)
library(scales)
library(ggplot2)
library(patchwork)
library(dplyr) 

#################### Kendall's Tau ####################
rank_matrix <- read_csv("rank_data.csv") %>% 
  column_to_rownames("Method")
cor_matrix <- cor(rank_matrix, method = "kendall", use = "pairwise.complete.obs")

cor_long <- as.data.frame(cor_matrix) %>%
  rownames_to_column("Metric1") %>%
  pivot_longer(
    cols = -Metric1,
    names_to = "Metric2",
    values_to = "Correlation"
  ) %>%
  mutate(
    Metric1 = factor(Metric1, levels = colnames(rank_matrix)),
    Metric2 = factor(Metric2, levels = colnames(rank_matrix))
  )
#
metric_order<- c("PCC", "Cos_Sim", "SSIM", "MI")
metric_order_new<- c( "Average \nAccuracy"," Overlap \n Ratio \n Layer2", "Overlap \n Ratio \n Layer3"
                        ,"Overlap \n Ratio \n Layer4" ,"Overlap \n Ratio \n Layer5"
                        ,"Overlap \n Ratio \n Layer6", "Overlap \n Ratio \n WM" )
#
all_metric_order<-c(metric_order,metric_order_new) 
label_mapping<-c("PCC","Cos_Sim","SSIM","MI",
                 "Average Accuracy"," Overlap Ratio Layer2","Overlap Ratio Layer3",
                 "Overlap Ratio Layer4","Overlap Ratio Layer5", 
                 "Overlap Ratio Layer6","Overlap Ratio WM" )

blue_red_palette <- colorRampPalette(c( "#982B2D","#C84747","#EE9D9F","#FCCDC9","#f7f7f7","#f7f7f7","#f7f7f7","#BBE6FA","#89CAEA","#0B75B3","#015696" ))(100)

cor_long <- cor_long %>%
  mutate(
    x_rank = as.numeric(factor(Metric1, levels = all_metric_order)),
    y_rank = as.numeric(factor(Metric2, levels = all_metric_order)),
    is_lower = x_rank > y_rank
  )

kendall_heatmap <- ggplot() +

  geom_tile(
    data = filter(cor_long, !is_lower),
    aes(x = factor(Metric1, levels = all_metric_order), 
        y = factor(Metric2, levels = rev(all_metric_order)), 
        fill = Correlation),
    color = "grey90", linewidth = 0.5
  ) +

  geom_tile(
    data = filter(cor_long, is_lower),
    aes(x = factor(Metric1, levels = all_metric_order),
        y = factor(Metric2, levels = rev(all_metric_order))),
    fill = "white", color = "grey90", linewidth = 0.5
  ) +

  geom_text(
    data = filter(cor_long, is_lower),
    aes(x = factor(Metric1, levels = all_metric_order),
        y = factor(Metric2, levels = rev(all_metric_order)),
        label = round(Correlation, 2), 
        color =pmin(Correlation + 0.3, 1)),
    size = 3
  ) +
  scale_fill_gradientn(
    colours = blue_red_palette,
    limits = c(-1, 1),
    name = "Kendall's tau \ncorrelation",
    guide = guide_colorbar(
      barwidth = unit(0.5, "cm"),
      barheight = unit(1.7, "cm"),
      title.position = "top"
    )
  ) +
  scale_color_gradientn(
    colours = blue_red_palette,
    limits = c(-1, 1),
    guide = "none"
  ) +
  scale_x_discrete(
    position = "bottom",
    labels = label_mapping,
    expand = c(0, 0)
  ) +
  scale_y_discrete(
    labels = rev(label_mapping),
    expand = c(0, 0)
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = "Agreement on ranking"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1,
      margin = margin(t = 7, b = 10)
    ),
    axis.text.y = element_text(
      hjust = 1,
      margin = margin(r = 5)
    ),
    axis.ticks.length = unit(0, "pt")
  ) +
  coord_fixed()
