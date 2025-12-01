4 FinBERT–LSTM Based Smart Stock Prediction System

This report describes a production-oriented FinBERT–LSTM pipeline that combines sentiment extracted from financial news with historical monthly returns to predict the next month's stock return. The design emphasizes modularity, data-quality checks, and deployment readiness to support reproducible inference in both local and server environments.

4.1 Model Architecture

The system implements a two-stage architecture. First, a domain-specific Transformer (FinBERT) converts individual news headlines into per-article sentiment scores and these are aggregated to a monthly sentiment indicator. Second, a compact stacked LSTM consumes a rolling sequence of four monthly feature vectors—each vector containing the monthly return and the aggregated sentiment—and outputs a single scalar representing the predicted next-month return. The pipeline is modular so the NLP and time-series components can be updated or replaced independently, and feature normalization, missing-data handling, and input validation are applied before model inference.

4.2 FinBERT Sentiment Module

The FinBERT module handles text preprocessing (tokenization, truncation/padding, and attention masks), batched inference, and conversion from class logits to a signed continuous sentiment score (e.g., positive_prob - negative_prob). For each month, per-article scores are aggregated (simple mean by default, with median or trimmed-mean as options) to form a robust monthly sentiment feature. The implementation includes safeguards for months with sparse or no news (fallback to neutral score) and supports caching of recent article embeddings to reduce repeated inference cost.

4.3 LSTM Prediction Module

The LSTM predictor is compact and tuned for small-sample financial time series: two stacked LSTM layers (64 hidden units each), dropout between layers, and a linear regression head that maps the final timestep representation to a scalar return prediction. Inputs are standardized per feature and sequences shorter than four months are padded with appropriate neutral values. The model exposes a deterministic inference API that accepts batched sequences and returns both point estimates and optional uncertainty proxies (e.g., ensemble or MC-dropout outputs) for downstream decision logic.

4.4 Training & Evaluation

Training uses supervised learning with sliding windows: four-month input sequences with the following month’s return as target. Optimization uses the Adam optimizer and MSE loss, with L2 weight decay and dropout to mitigate overfitting. Model selection relies on time-aware validation (walk-forward or expanding-window splits) and metrics include MAE, RMSE, and directional accuracy (up/down). Robustness checks cover missing-news scenarios, alternative aggregation methods for sentiment, and sensitivity to input scaling; early stopping and checkpointing are used to preserve the best validation model.

4.5 API Design & Real-time System

The production service exposes a simple REST/HTTP endpoint that orchestrates data retrieval, feature assembly, sentiment scoring, and model inference. A typical request flow: (1) fetch historical monthly prices from the price API; (2) collect related news headlines for the corresponding months; (3) compute monthly sentiment via FinBERT; (4) assemble and normalize the 4-month input matrix; (5) run LSTM inference and return predicted return and derived price. The server includes caching layers, configurable timeouts and retries for upstream APIs, structured error responses, and logging/metrics for monitoring. Deployment-ready scripts and configuration (environment variables, model path) allow reproducible runs on cloud servers or local machines.

4.6 Conclusion

Combining FinBERT-derived sentiment with a compact LSTM time-series model produces a pragmatic and interpretable forecasting pipeline that improves responsiveness to new textual information while retaining temporal modeling of price dynamics. The modular design, built-in robustness measures, and API-oriented interfacing make this approach suitable for both experimental research and prototype production deployments.