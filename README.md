# NER

# *Available models*

  - [Model Description][model_description]
    
  - Model:
    
    - [How to train the **JulioCP model**][JulioCP_folder]

# *Main assumptions*

### - OLD model

- *Framework*: [**Java-Scala**]
    - **Pyspark 2.4.4** and **spark-nlp 2.5.3**
    - **Pyspark 3.1.1** and **spark-nlp 3.0.2** (This version has been tested by **JulioCP**)
- *Neural Network architecture*: 
    - **LSTMs** (Bert model from google)
- *Requirements*: 
    - **Training**: **500Gb-700Gb RAM** to train ~300.000 samples
    - **Prediction**: Depends on the memory **RAM** , and therefore, the amount of samples
- *Inference*: 
    - The inference will be performed by a **JAR** artifact, which call to the model trained (*Only tested with: Pyspark 2.4.4 and spark-nlp 2.5.3*).
    - This model **can not be serialized by ONNX** (directly).
  

### - JulioCP model

- *Framework*: [**Python**]
    - **torch 1.8.1**
- *Neural Network architecture*: 
    - **Transformers** (Bert model from google)
- *Requirements*: 
    - **Training**: Data will be taken according to the batch which is going to be trained
    - **Prediction**: Data will be taken according to the batch which is going to be predicted
- *Inference*:
    - This model can be serialized by ONNX directly.

[model_description]: ./JulioCP-model/description.md
[JulioCP_folder]: ./JulioCP-model/howtotrainJulioCP.md
