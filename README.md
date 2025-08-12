## Extreme Multilabel Classification
• Fine tuned **T5 Transformer(Google AI)** for **E commerce product categorisation** from concatenated input strings
• Built a hierarchical k-means clustering pipeline using **BERT** embeddings for scalable, tree-based label partitioning
• Designed **prefix coded label encoding** to help the Transformer distinguish and focus on tail labels in hierarchy
• Integrated **hierarchical clustering** into T5 pipeline, so model predicts prefix code sequences for cluster positions
• Recorded an **6%** increase in accuracy from Plain T5 **(accuracy: 79%)** to Hierarchical T5 **(accuracy : 85%)**
