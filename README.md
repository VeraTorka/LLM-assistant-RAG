# LLM-assistant-RAG
This is your friendly assistant which is RAG-application built as part of LLM-zoomcamp'25


   
Evaluation Criteria

Problem description

Retrieval flow


Retrieval evaluation
The basic approach - using minsearch without any boosting - gave the following metrics:

{'hit_rate': 0.6471774193548387, 'mrr': 0.5518553187403998}
The improved version (with tuned boosting):

{'hit_rate': 0.8891129032258065, 'mrr': 0.7286029505888372}
The best boosting parameters:

boost = {
    'food': 2.84,
     'serving_size_g': 1.97,
     'calories_kcal': 2.03,
     'protein_g': 0.91,
     'fat_g': 1.89,
     'carbohydrates_g': 2.19,
     'vitamin_a_mg': 0.72,
     'vitamin_b6_mg': 1.62,
     'vitamin_b12_mg': 0.77,
     'vitamin_c_mg': 0.09,
     'vitamin_d_mg': 2.45,
     'vitamin_e_mg': 2.42,
     'calcium_mg': 0.47,
     'iron_mg': 0.21,
     'potassium_mg': 2.00,
     'magnesium_mg': 1.45,
     'selenium_mg': 1.41,
     'zinc_mg': 2.64,
     'iodine_mg': 0.62,
     'allergens': 2.64,
}

LLM evaluation

RAG flow evaluation
We used the LLM-as-a-Judge metric to evaluate the quality of our RAG flow.

For gpt-4o-mini, in a sample with 200 records, we had:
relevance
RELEVANT           129
NON_RELEVANT        43
PARTLY_RELEVANT     28
Name: count, dtype: int64

RELEVANT           0.660
NON_RELEVANT       0.175
PARTLY_RELEVANT    0.165
Name: proportion, dtype: float64

We also tested gpt-4o:

relevance
RELEVANT           148
PARTLY_RELEVANT     37
NON_RELEVANT        15
Name: count, dtype: int64
relevance
RELEVANT           0.740
PARTLY_RELEVANT    0.185
NON_RELEVANT       0.075
Name: proportion, dtype: float64
The difference is minimal, so we opted for gpt-4o-mini.



Interface


Ingestion pipeline


Monitoring



Containerization


Reproducibility



Best practices
 Hybrid search: combining both text and vector search (at least evaluating it) (1 point)
 Document re-ranking (1 point)
 User query rewriting (1 point)
Bonus points (not covered in the course)
 Deployment to the cloud (2 points)
 Up to 3 extra bonus points if you want to award for something extra (write in feedback for what)