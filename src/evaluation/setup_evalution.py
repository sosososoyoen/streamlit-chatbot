from langsmith import Client
import json
import os
from dotenv import load_dotenv
import openai
from langsmith import wrappers
import streamlit as st

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "testset.json")
with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Define the input and reference output pairs that you'll use to evaluate your app
client = Client()
dataset_name = "QA Example Dataset"
# Create the dataset
# dataset = client.create_dataset(
#     dataset_name=dataset_name, description="A sample dataset in LangSmith."
# )

# Create examples in the dataset. Examples consist of inputs and reference outputs
examples = [
    {
        "inputs": {"question": item["question"]},
        "outputs": {"answer": item["answer"]},
    }
    for item in data
]

# Add the examples to the dataset
# client.create_examples(dataset_id=dataset.id, examples=examples)

openai_client = wrappers.wrap_openai(openai.OpenAI(api_key=st.secrets["OPENAI_KEY"]))

eval_instructions = "You are an expert professor specialized in grading students' answers to questions."


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    user_content = f"""You are grading the following question:
{inputs['question']}
Here is the real answer:
{reference_outputs['answer']}
You are grading the following predicted answer:
{outputs['response']}
Respond with CORRECT or INCORRECT:
Grade:
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content
    return response == "CORRECT"


def concision(outputs: dict, reference_outputs: dict) -> bool:
    return len(outputs["response"]) < 2 * len(reference_outputs["answer"])


default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."


def my_app(question: str, model: str = "gpt-4o-mini", instructions: str = default_instructions) -> str:
    return openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content


def ls_target(inputs: dict) -> dict:
    return {"response": my_app(inputs["question"])}


def ls_target_v2(inputs: dict) -> dict:
    return {"response": my_app(inputs["question"], model="gpt-4o")}


# experiment_results = client.evaluate(
#     ls_target,
#     dataset_name,
#     evaluators=[concision, correctness],
#     experiment_prefix="openai-4o-mini",
# )

experiment_results_v2 = client.evaluate(
    ls_target_v2,
    data=dataset_name,
    evaluators=[concision, correctness],
    experiment_prefix="openai-4-turbo",
)

instructions_v3 = ("Respond to the users question in a short, concise manner (one short sentence). Do NOT use more "
                   "than ten words.")


def ls_target_v3(inputs: dict) -> dict:
    response = my_app(
        inputs["question"],
        model="gpt-4-turbo",
        instructions=instructions_v3
    )
    return {"response": response}


experiment_results_v3 = client.evaluate(
    ls_target_v3,
    dataset_name,
    evaluators=[concision, correctness],
    experiment_prefix="strict-openai-4-turbo",
)
