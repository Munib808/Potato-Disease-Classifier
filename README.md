### **Potato Disease Classification | Deep Learning & Computer Vision**
This project uses a MobileNet-based CNN to automatically classify potato leaf images into Early Blight, Late Blight, and Healthy categories. The model is trained on labeled image data and deployed as a web application, enabling fast, accurate, and accessible disease detection for agricultural support.

* **Data Engineering:** Processed and optimized a dataset of **2,152 images** from the PlantVillage collection, implementing `tf.keras.preprocessing` for efficient data loading and batching.

* **Architecture Comparison:** Evaluated and implemented three distinct deep learning models to find the optimal balance between accuracy and computational efficiency:

    * **Custom CNN:** A sequential model tailored for specific leaf features.
    * **MobileNetV2:** Utilized for its lightweight architecture, ideal for potential mobile deployment.
    * **ResNet50:** Leveraged deep residual learning to achieve high-performance feature extraction.
 
* **Model Training & Optimization:** Standardized input dimensions to **256x256 pixels** and trained models over **50 epochs** with a batch size of 32 to ensure convergence and robust pattern recognition.

* **Deployment Readiness:** Successfully exported and saved three production-ready models in `.keras` format for integration into web or mobile applications.

* **Tech Stack:** Python, TensorFlow, Keras, NumPy, Matplotlib.
