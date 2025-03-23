from document_retrieval import DocumentRetriever
from datasets import load_dataset
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm


def format_user_prompt(question: str, choices: List[str], docs: List[Dict[str, Any]]) -> str:
    """
    Format user prompt.

    Args:
        question: The question text
        choices: The multiple choice options
        docs: The relevant wiki docs
    
    Returns:
        Formatted user prompt for instruct model
    """
    formatted_docs = ''.join([f"""Title: {doc['title']}\n{doc['text']}""" for doc in docs])
    formatted_choices = '\n'.join(choices)
    prompt = f"""
You are given the following documents from Wikipedia:
{formatted_docs}

Based on the above information, answer the following multiple choice question:
{question}\n
{formatted_choices}
"""
    return prompt

def extract_answer(resp: str) -> str:
    """
    Extract multiple choice answer from model response.

    Args:
        resp: Model response to some question
    
    Returns:
        Multiple choice answer
    """
    if 'A' in resp:
        return 'A'
    if 'B' in resp:
        return 'B'
    if 'C' in resp:
        return 'C'
    if 'D' in resp:
        return 'D'
    
    return None


if __name__ == "__main__":

    # Load datasets
    wiki = load_dataset("nuprl/engineering-llm-systems", 
                        name="wikipedia-northeastern-university", 
                        split="test")
    
    questions = load_dataset("nuprl/engineering-llm-systems", 
                            name="obscure_questions", 
                            split="tiny")

    # Create document retriever
    retriever = DocumentRetriever(wiki)

    # Load model client
    client = OpenAI(
        api_key="fahy.jo@northeastern.edu:46582",
        base_url="https://nerc.guha-anderson.com/v1"
    )

    # Answer questions
    num_correct = 0
    solved_questions = []
    for (i, question) in tqdm(enumerate(questions), total=len(questions)):
        prompt = question['prompt']
        choices = question['choices']
        answer = question['correct_answer']

        # TF-IDF retrieval / Neural reranking
        docs = retriever.retrieve_documents(prompt, final_top_n=8)

        # Format messages
        user_prompt = format_user_prompt(prompt, choices, docs)
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        # Query model
        resp = client.chat.completions.create(
            messages = messages,
            model = "llama3p1-8b-instruct",
            temperature=0
        )
        
        # Extract answer
        model_answer = extract_answer(resp.choices[0].message.content)
        if not model_answer:
            print(f"Failed to extract answer for question: {i}")
        
        # Check answer
        if model_answer == answer:
            num_correct += 1
            solved_questions.append(i)
    
    # Print results
    print(f"Solved {num_correct} questions")
    print(f"Solved questions: {solved_questions}")