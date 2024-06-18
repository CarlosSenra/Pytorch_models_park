import os 

PATH_DATASETS = os.path.join(os.path.join(os.getcwd(), "EDA"),"dataframe_model")


PATH_PREDICTIONS_GLU = os.path.join(os.path.join(os.getcwd(), "Prediction_Values"),"GLU")
PATH_METRICS_VALUES_GLU = os.path.join(os.path.join(os.getcwd(), "Evaluation_Metrics"),"GLU")

PATH_PREDICTIONS_LSTM = os.path.join(os.path.join(os.getcwd(), "Prediction_Values"),"LSTM")
PATH_METRICS_VALUES_LSTM = os.path.join(os.path.join(os.getcwd(), "Evaluation_Metrics"),"LSTM")


PATH_PREDICTIONS_TFT = os.path.join(os.path.join(os.getcwd(), "Prediction_Values"),"TFT")
PATH_PREDICTIONS_TFT_SELECT_FEATURES = os.path.join(os.path.join(os.path.join(os.getcwd(), "Prediction_Values"),"TFT"),"select_feature")

PATH_METRICS_VALUES_TFT = os.path.join(os.path.join(os.getcwd(), "Evaluation_Metrics"),"TFT")
PATH_METRICS_VALUES_TFT_SELECT_FEATURES = os.path.join(os.path.join(os.path.join(os.getcwd(), "Evaluation_Metrics"),"TFT"),"select_feature")

PATH_ATTENTION = os.path.join(os.path.join(os.getcwd(), "Attention_values"),"TFT_attention")
PATH_ATTENTION_SELECT_FEATURES = os.path.join(os.path.join(os.getcwd(), "Attention_values"),"Feature_selection")