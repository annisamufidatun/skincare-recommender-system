# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

![skincare(s)](https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1000w,f_avif,q_auto:eco,dpr_2/rockcms/2024-01/240117-staff-skin-care-routines-social-2c85d8.jpg)

This project aims to develop a skincare product recommender system using both content-based filtering and collaborative filtering approaches. The importance of this project lies in the growing skincare industry and the need for personalized product recommendations to help consumers navigate the vast array of available products.

The global skincare market has seen significant growth, with its value projected to reach $189.3 billion by 2025 [1]. This rapid expansion has led to an overwhelming variety of products, making it challenging for consumers to find suitable options for their specific needs. Recommender systems have emerged as a crucial tool in e-commerce, with studies showing they can significantly increase sales and customer satisfaction [2].

Skincare is a highly personal and often complex topic, with different products working differently for various skin types and conditions. A recommender system can significantly enhance the user experience by suggesting products that are likely to suit an individual's needs, based on product characteristics and user preferences. Research has shown that personalized recommendations in the beauty and skincare industry can lead to increased customer engagement and loyalty. Moreover, the integration of artificial intelligence and machine learning in skincare recommendations has been gaining traction. A study by Hwang et al. (2020) demonstrated that AI-powered skincare recommendation systems can provide more accurate and personalized suggestions compared to traditional methods. By developing a hybrid recommender system that combines content-based and collaborative filtering approaches, this project aims to address the limitations of single-method systems and provide more robust and accurate recommendations [3].

**References**

[1] Grand View Research. (2019). Skin Care Products Market Size, Share & Trends Analysis Report By Product (Face Cream, Body Lotion), By Region (North America, Central & South America, Europe, APAC, MEA), And Segment Forecasts, 2019 - 2025. https://www.grandviewresearch.com/industry-analysis/skin-care-products-market

[2] Schafer, J. B., Konstan, J. A., & Riedl, J. (2001). E-commerce recommendation applications. Data Mining and Knowledge Discovery, 5(1-2), 115-153. https://link.springer.com/article/10.1023/A:1009804230409

[3] Burke, R. (2002). Hybrid recommender systems: Survey and experiments. User modeling and user-adapted interaction, 12(4), 331-370. https://link.springer.com/article/10.1023/A:1021240730564

## Business Understanding


### Problem Statements

- How can we effectively recommend skincare products to users based on product categories and characteristics?
- How can we leverage user ratings and interactions to provide personalized product recommendations?
- How can we combine both product features and user behavior to create a comprehensive recommendation system?

### Goals

- Develop a content-based filtering system that recommends products based on their categories and similarities.
- Create a collaborative filtering system that provides personalized recommendations based on user ratings and behaviors.
- Implement and compare both recommendation approaches to provide a robust and effective skincare product recommendation system.


### Solution Approach
- Content-Based Filtering: Utilize TF-IDF and cosine similarity to recommend products based on their categories.
- Collaborative Filtering: Implement a neural network model to predict user ratings for products they haven't tried yet.

## Data Understanding
This dataset was collected via Python scraper in March 2023 and contains:
- information about all beauty products (8494 products) from the Sephora online store, including product and brand names, prices, ingredients, ratings, and all features.
- user reviews (about 1 million on over 2,000 products) of all products from the Skincare category, including user appearances, and review ratings by other users

[Link to dataset.](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

### Variabel-variabel pada dataset adalah sebagai berikut

**Product Dataset**

| Feature             | Description                                                                                                   |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| `product_id`        | The unique identifier for the product from the site                                                          |
| `product_name`      | The full name of the product                                                                                  |
| `brand_id`          | The unique identifier for the product brand from the site                                                     |
| `brand_name`        | The full name of the product brand                                                                            |
| `loves_count`       | The number of people who have marked this product as a favorite                                              |
| `rating`            | The average rating of the product based on user reviews                                                       |
| `reviews`           | The number of user reviews for the product                                                                   |
| `size`              | The size of the product, which may be in oz, ml, g, packs, or other units depending on the product type       |
| `variation_type`    | The type of variation parameter for the product (e.g. Size, Color)                                            |
| `variation_value`   | The specific value of the variation parameter for the product (e.g. 100 mL, Golden Sand)                      |
| `variation_desc`    | A description of the variation parameter for the product (e.g. tone for fairest skin)                         |
| `ingredients`       | A list of ingredients included in the product, for example: [‘Product variation 1:’, ‘Water, Glycerin’, ‘Product variation 2:’, ‘Talc, Mica’] or if no variations [‘Water, Glycerin’] |
| `price_usd`         | The price of the product in US dollars                                                                         |
| `value_price_usd`   | The potential cost savings of the product, presented on the site next to the regular price                    |
| `sale_price_usd`    | The sale price of the product in US dollars                                                                    |
| `limited_edition`   | Indicates whether the product is a limited edition or not (1-true, 0-false)                                  |
| `new`               | Indicates whether the product is new or not (1-true, 0-false)                                                |
| `online_only`       | Indicates whether the product is only sold online or not (1-true, 0-false)                                  |
| `out_of_stock`      | Indicates whether the product is currently out of stock or not (1 if true, 0 if false)                       |
| `sephora_exclusive` | Indicates whether the product is exclusive to Sephora or not (1 if true, 0 if false)                         |
| `highlights`        | A list of tags or features that highlight the product's attributes (e.g. [‘Vegan’, ‘Matte Finish’])           |
| `primary_category`  | First category in the breadcrumb section                                                                      |
| `secondary_category`| Second category in the breadcrumb section                                                                     |
| `tertiary_category` | Third category in the breadcrumb section                                                                      |
| `child_count`       | The number of variations of the product available                                                             |
| `child_max_price`   | The highest price among the variations of the product                                                          |
| `child_min_price`   | The lowest price among the variations of the product                                                           |

**Reviews Dataset**

| Feature                | Description                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------------|
| `author_id`            | The unique identifier for the author of the review on the website                                          |
| `rating`               | The rating given by the author for the product on a scale of 1 to 5                                        |
| `is_recommended`       | Indicates if the author recommends the product or not (1-true, 0-false)                                    |
| `helpfulness`          | The ratio of all ratings to positive ratings for the review: helpfulness = total_pos_feedback_count / total_feedback_count |
| `total_feedback_count` | Total number of feedback (positive and negative ratings) left by users for the review                      |
| `total_neg_feedback_count` | The number of users who gave a negative rating for the review                                               |
| `total_pos_feedback_count` | The number of users who gave a positive rating for the review                                               |
| `submission_time`      | Date the review was posted on the website in the 'yyyy-mm-dd' format                                        |
| `review_text`          | The main text of the review written by the author                                                            |
| `review_title`         | The title of the review written by the author                                                               |
| `skin_tone`            | Author's skin tone (e.g. fair, tan, etc.)                                                                    |
| `eye_color`            | Author's eye color (e.g. brown, green, etc.)                                                                |
| `skin_type`            | Author's skin type (e.g. combination, oily, etc.)                                                            |
| `hair_color`           | Author's hair color (e.g. brown, auburn, etc.)                                                              |
| `product_id`           | The unique identifier for the product on the website                                                        |


### **Exploratory Data Analysis**

To understand the data, several methods were employed:

- Examining the size of the dataset

- Viewing the data type of each feature using **data.info()**

- Observing statistical information of numeric features using **data.describe()**

- Checking for null values in the dataset using **data.isnull().sum()**

- Checking for duplicate values in the dataset using **data.duplicated().sum()**

- Visualizing the top 5 brands and ratings from the product and reviews datasets

![top 5 brand](https://raw.githubusercontent.com/annisamufidatun/skincare-recommender-system/0c7ac70c32e6af41081023da4b7b28bad2753efe/assests/top_5_brand.png)

![rating produk](https://raw.githubusercontent.com/annisamufidatun/skincare-recommender-system/0c7ac70c32e6af41081023da4b7b28bad2753efe/assests/rating_product.png)

![top 5 brand](https://raw.githubusercontent.com/annisamufidatun/skincare-recommender-system/0c7ac70c32e6af41081023da4b7b28bad2753efe/assests/review_product.png)

This approach provides a comprehensive overview of the dataset structure, content, and quality, helping to identify any potential issues or patterns in the data before proceeding with further analysis and modeling.


## Data Preparation

1. **Filtered the product dataset to include only skincare products:**
   The initial product dataset contains various categories. To focus on skincare, the dataset was filtered to retain only entries where the primary category is 'Skincare'. This step narrows down the data to products relevant to the skincare domain, ensuring that subsequent analyses and recommendations are specific to skincare products.

2. **Filled null values in tertiary_category with values from secondary_category:**
  The `tertiary_category` column have missing values. To maintain consistency and avoid gaps in data, these null values were filled with corresponding values from the `secondary_category`. This ensures that each product has a complete category assignment, which is important for accurate analysis and recommendations.

3. **Created a subset of the product dataset with relevant features: 'product_id', 'product_name', 'brand_name', 'tertiary_category' (renamed to 'category'), and 'rating':**
  To simplify the analysis and focus on essential attributes, a new DataFrame was created that includes only the most relevant columns: `product_id`, `product_name`, `brand_name`, `tertiary_category` (renamed to `category`), and `rating`. This subset is used for further analysis and recommendation purposes.

4. **For the reviews dataset, dropped rows with null values and selected relevant features: 'author_id', 'rating', 'product_id':**
  In the reviews dataset, rows with missing values were removed to ensure data quality. The remaining dataset was then filtered to include only the columns `author_id`, `rating`, and `product_id`, which are crucial for analyzing user reviews and performing collaborative filtering.

5. **Merged product and review datasets, ensuring only reviews with corresponding product_id in the product dataset are included:**
  The product and review datasets were combined using `product_id` to ensure that only reviews associated with valid products are included. This step integrates product information with user reviews, allowing for a comprehensive dataset that links user feedback to product details.

6. **Created a subset of 100,000 rows while maintaining diversity in product_id values:**
  To manage the size of the dataset, a subset of 100,000 rows was created. This subset includes a diverse set of product IDs, helping to avoid overrepresentation of any single product and ensuring the dataset is suitable for training and evaluation.

7. **Performed encoding of user_id and product_id for the collaborative filtering model:**
  The `user_id` and `product_id` columns were encoded into numeric values to facilitate processing by machine learning algorithms. This encoding transforms categorical identifiers into numerical representations, which is necessary for training the collaborative filtering model.

8. **Normalized ratings to a 0-1 scale for the collaborative filtering model:**
  Ratings were normalized to a 0-1 scale to standardize the input for the collaborative filtering model. This normalization adjusts the ratings so that they fall within a consistent range, making the training process more stable and effective.

9. **Split the data into training and validation sets (80% train, 20% validation):**
  The dataset was divided into training and validation sets to evaluate the performance of the collaborative filtering model. 80% of the data was used for training the model, while the remaining 20% was reserved for validation. This split helps assess the model's ability to generalize to unseen data and ensures that it is not overfitting to the training set.



## Modeling

### **Content-Based Filtering**


**Strengths:**
1. Independent of Other Users: Does not rely on data from other users, making it suitable for systems with limited user data.
2. Specific Recommendations: Provides recommendations based on specific features of the products, tailored to individual user preferences.
3. Handles New Products: Can recommend new products that have not been widely reviewed since the analysis is based on product features.

**Weaknesses:**
1. Over-Specialization: Recommendations are often limited to products similar to those previously rated by the user, reducing recommendation diversity.
2. Ignores Other Users’ Preferences: Does not take into account the preferences or reviews of other users, which might not always reflect the overall quality of products.
3. Requires Relevant Features: Needs descriptive features for each product, which may not always be available or sufficiently detailed.

**Steps:**

1. **Used TF-IDF Vectorization on the 'category' Column**
   - Applied the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique to the 'category' column of the products dataset.
   - TF-IDF converts text into numerical vectors that reflect the importance of each term in a document relative to a corpus.

2. **Computed Cosine Similarity Between Products**
   - Calculated the cosine similarity between products using their TF-IDF vectors.
   - Cosine similarity measures the cosine of the angle between two vectors, providing a measure of similarity based on the features.

3. **Implemented a Function to Recommend Products Based on Similarity Scores**
   - Created a function that generates product recommendations by finding the most similar products based on cosine similarity scores.
   - This function returns a list of recommended products similar to a given product, excluding the product itself.

### **Collaborative Filtering**


**Strengths:**
1. Community-Based Recommendations: Utilizes reviews and behavior of other users to provide recommendations that may be more relevant and diverse.
2. Captures Complex Preferences: Can capture complex, non-obvious user-product relationships and preferences.
3. No Need for Product Information: Does not require descriptive features of products; relies solely on user interactions and reviews.

**Weaknesses:**
1. Data Sparsity Issues: Performance can suffer if there is limited data for users or products, leading to the cold-start problem where new or less popular items are hard to recommend.
2. Scalability: Can be computationally intensive on large datasets, particularly with a high number of users and products.
3. Ignores Product Features: Does not consider the attributes or features of products, which could lead to less personalized recommendations.


**Steps:**

1. **Developed a Neural Network Model (RecommenderNet) Using Keras**
   - Designed a custom neural network model named `RecommenderNet` using the Keras library.
   - The model incorporates embeddings for users and products to learn latent features that represent user preferences and product characteristics.

2. **Model Uses Embedding Layers for Users and Products**
   - Employed embedding layers to convert user and product IDs into dense vectors.
   - These embeddings capture the latent relationships between users and products, allowing the model to learn effective representations.

3. **Computes the Dot Product of Embeddings and Bias Terms**
   - The model computes the dot product between user and product embeddings to estimate interaction scores.
   - Added user and product biases to the dot product to capture additional variations in user preferences and product characteristics.

4. **Applied Sigmoid Activation to the Output**
   - Applied a sigmoid activation function to the output of the model to map the predicted ratings to a range between 0 and 1.
   - This helps in predicting ratings that are bounded within a certain range.

5. **Trained the Model for 100 Epochs with a Batch Size of 128**
   - Trained the neural network model using a dataset split into training and validation sets.
   - The model was trained for 100 epochs with a batch size of 128, optimizing the weights to minimize the binary cross-entropy loss function.



## Evaluation


### **Content-Based Filtering**

The content-based filtering system was evaluated both qualitatively and quantitatively:

1. Qualitative Evaluation:
   The system's performance was assessed by examining recommendations for sample products. It successfully recommends products from similar categories, demonstrating its ability to capture product similarities based on their attributes.

2. Quantitative Evaluation:
   We used the Recall@K metric to quantitatively evaluate the system's performance. In this case, we set K=5, meaning we considered the top 5 recommendations.

   - Predicted Products: The system's top 5 recommended products.
   - Relevant Products: The actual products that are considered relevant or correct recommendations.

   We calculated the Recall@5 using the following formula:
   $$
    \text{Recall@5} = \frac{\text{Number of relevant products in top 5 recommendations}}{\text{Total number of relevant products}}
   $$

   Our code calculates this metric, providing a percentage that indicates how many of the relevant products were successfully included in the top 5 recommendations. The resulting Recall@5 score gives us a quantitative measure of the system's ability to recommend relevant products within its top recommendations. The result of the Recall@5 is **100%**.

### **Collaborative Filtering**

For this method, I use RMSE or Root Mean Squared Error. RMSE is a commonly used metric for evaluating the accuracy of prediction models, particularly in recommendation systems. The formula for RMSE is:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_{\text{pred},i} - y_{\text{true},i})^2}
$$

Where:
- y_pred is the predicted rating
- y_true is the actual rating
- n is the number of predictions

A lower RMSE indicates better model performance. In our case, the final validation RMSE of 0.3540 suggests that, on average, our predictions deviate from the true ratings by about 0.3540 units on our rating scale. This metric provides a clear, interpretable measure of the model's prediction accuracy, allowing for easy comparison between different model iterations or against benchmark performance levels.

![model result](https://raw.githubusercontent.com/annisamufidatun/skincare-recommender-system/0c7ac70c32e6af41081023da4b7b28bad2753efe/assests/model_result.png)

The collaborative filtering model's performance was assessed using Root Mean Squared Error (RMSE) over 50 epochs. The final epoch yielded a training RMSE of 0.2591 and a validation RMSE of 0.3540. The training curve shows steady improvement, indicating effective learning from the dataset. However, the validation curve exhibits high volatility, suggesting inconsistent performance on unseen data. This volatility, combined with the persistent gap between training and validation RMSE, points to potential overfitting. While the model demonstrates good performance on training data, its generalization capabilities may be limited. The final validation RMSE of 0.3540 indicates moderate predictive accuracy, but there's room for improvement. Future iterations could focus on regularization techniques or architectural adjustments to enhance the model's stability and generalization. Despite these challenges, the model shows promise in capturing user-item interactions, forming a foundation for a recommendation system that can be further refined for better real-world application.


## Conclusion

In conclusion, this project successfully implemented both content-based and collaborative filtering approaches to create a comprehensive skincare product recommender system. The content-based system provides recommendations based on product similarities, while the collaborative filtering system offers personalized recommendations based on user behavior. Future work could focus on combining these approaches and incorporating additional features to further improve recommendation quality.
