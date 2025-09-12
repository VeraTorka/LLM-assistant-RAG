#!/usr/bin/env python
# coding: utf-8

# In[413]:


##Data Set Ingestion and Cleaning


# In[490]:


import pandas as pd


# In[491]:


df=pd.read_csv('../data/data.csv')


# In[492]:


##Data Set Indexing


# In[493]:


get_ipython().system(' wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/refs/heads/main/minsearch.py')


# In[494]:


documents=df.to_dict(orient='records')


# In[495]:


cleaned_docs = []
for doc in documents:
    clean_doc = {}
    for k, v in doc.items():
        if v is None:  # None → пустая строка или 0
            clean_doc[k] = "" if isinstance(v, str) else 0
        elif isinstance(v, float):
            import math
            clean_doc[k] = "" if math.isnan(v) else v
        else:
            clean_doc[k] = v
    cleaned_docs.append(clean_doc)

documents = cleaned_docs


# In[461]:


import minsearch


# In[496]:


index = minsearch.Index(
    text_fields=['food', 'serving_size_g', 'calories_kcal', 'protein_g', 'fat_g',
       'carbohydrates_g', 'vitamin_a_mg', 'vitamin_b6_mg', 'vitamin_b12_mg',
       'vitamin_c_mg', 'vitamin_d_mg', 'vitamin_e_mg', 'calcium_mg', 'iron_mg',
       'potassium_mg', 'magnesium_mg', 'selenium_mg', 'zinc_mg', 'iodine_mg',
       'allergens'],
    keyword_fields=['id']
)


# In[497]:


index.fit(documents)


# In[498]:


##RAG Flow


# In[499]:


import os


# In[500]:


from openai import OpenAI
client = OpenAI()


# In[501]:


def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,

    )

    return results


# In[502]:


prompt_template = """
You are a nutrition assistant. Answer the QUESTION based on the CONTEXT from the food database. 
QUESTION: {question}
CONTEXT: {context}
""".strip()

entry_template = """
food: {food}
serving_size_g: {serving_size_g}
calories_kcal: {calories_kcal}
protein_g: {protein_g}
fat_g: {fat_g}
carbohydrates_g: {carbohydrates_g}
vitamin_a_mg: {vitamin_a_mg}
vitamin_b6_mg: {vitamin_b6_mg}
vitamin_b12_mg: {vitamin_b12_mg}
vitamin_c_mg: {vitamin_c_mg}
vitamin_d_mg: {vitamin_d_mg}
vitamin_e_mg: {vitamin_e_mg}
calcium_mg: {calcium_mg}
iron_mg: {iron_mg}
potassium_mg: {potassium_mg}
magnesium_mg: {magnesium_mg}
selenium_mg: {selenium_mg}
zinc_mg: {zinc_mg}
iodine_mg: {iodine_mg}
allergens: {allergens}
""".strip()


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[503]:


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# In[504]:


def rag(query, model='gpt-4o-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    #print(prompt)
    answer = llm(prompt, model=model)
    return answer


# In[505]:


question = 'Which meat has no allergen and have less calories'
answer = rag(question)
print(answer)


# In[472]:


##Retrieval evaluation


# In[506]:


df_question = pd.read_csv('../data/ground-truth-retrieval.csv')


# In[507]:


df_question.head()


# In[508]:


ground_truth = df_question.to_dict(orient='records')


# In[509]:


ground_truth[0]


# In[510]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[511]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[512]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


# In[513]:


from tqdm.auto import tqdm


# In[514]:


evaluate(ground_truth, lambda q: minsearch_search(q['question']))


# In[515]:


##Finding the best parameters


# In[517]:


df_question


# In[525]:


df_validation = df_question[:100]
df_test = df_question[100:]


# In[526]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)

        # Evaluate the objective function
        current_score = objective_function(current_params)

        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params

    return best_params, best_score


# In[527]:


gt_val = df_validation.to_dict(orient='records')


# In[528]:


evaluate(gt_val, lambda q: minsearch_search(q['question']))


# In[529]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[530]:


documents[0]


# In[531]:


param_ranges = {
     'food': (0.0, 3.0),
     'serving_size_g': (0.0, 3.0),
     'calories_kcal': (0.0, 3.0),
     'protein_g': (0.0, 3.0),
     'fat_g': (0.0, 3.0),
     'carbohydrates_g': (0.0, 3.0),
     'vitamin_a_mg': (0.0, 3.0),
     'vitamin_b6_mg': (0.0, 3.0),
     'vitamin_b12_mg': (0.0, 3.0),
     'vitamin_c_mg': (0.0, 3.0),
     'vitamin_d_mg': (0.0, 3.0),
     'vitamin_e_mg': (0.0, 3.0),
     'calcium_mg': (0.0, 3.0),
     'iron_mg': (0.0, 3.0),
     'potassium_mg': (0.0, 3.0),
     'magnesium_mg': (0.0, 3.0),
     'selenium_mg': (0.0, 3.0),
     'zinc_mg': (0.0, 3.0),
     'iodine_mg': (0.0, 3.0),
     'allergens': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[532]:


simple_optimize(param_ranges, objective, n_iterations=20)


# In[533]:


def minsearch_improved(query):
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

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

evaluate(ground_truth, lambda q: minsearch_improved(q['question']))


# In[ ]:


##RAG evaluation


# In[534]:


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[535]:


len(ground_truth)


# In[540]:


record = ground_truth[0]
question = record['question']
answer_llm=rag(question)


# In[541]:


print(answer_llm)


# In[542]:


prompt = prompt2_template.format(question=question, answer_llm=answer_llm)
print(prompt)


# In[543]:


import json


# In[544]:


df_sample = df_question.sample(n=200, random_state=1)


# In[545]:


sample = df_sample.to_dict(orient='records')


# In[556]:


evaluations = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question) 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[557]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[558]:


df_eval.relevance.value_counts()


# In[548]:


df_eval.relevance.value_counts(normalize=True)


# In[549]:


df_eval.to_csv('../data/rag-eval-gpt-4o-mini.csv', index=False)


# In[550]:


df_eval[df_eval.relevance == 'NON_RELEVANT']


# In[551]:


evaluations_gpt4o = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question, model='gpt-4o') 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations_gpt4o.append((record, answer_llm, evaluation))


# In[552]:


df_eval = pd.DataFrame(evaluations_gpt4o, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[553]:


df_eval.relevance.value_counts()


# In[554]:


df_eval.relevance.value_counts(normalize=True)


# In[559]:


df_eval.to_csv('../data/rag-eval-gpt-4o.csv', index=False)


# In[ ]:




