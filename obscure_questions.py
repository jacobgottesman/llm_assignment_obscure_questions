from document_retrieval import DocumentRetriever
from datasets import load_dataset
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm


def format_user_prompt(question: str, choices: List[str], docs: List[Dict[str, str]]) -> str:
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
    Extract the first multiple choice answer (A, B, C, D) that appears in a text.

    Args:
        resp: Model response to some question
    
    Returns:
        The first capital letter A, B, C, or D that appears in the text, or None if none found
    """
    for char in resp:
        if char in 'ABCD':
            return char
        
    
    return None

def solve(client, question: str, choices: List[str], documents: Dict[str, str]) -> str:
    """
    Answers a multiple choice question using retrieval augmented generation.

    `client` is the model api client.

    `question` is the text of the question.
    
    `choices` is the list of choices with leading letters. For example:

     ```
     ["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"]
     ```

    `documents` is the list of documents to use for retrieval augmented
    generation.

    The result should be the just the letter of the correct choice, e.g.,
    `"A"` but not `"A."` and not `"A. Choice 1"`.
    """
    # Retrieve documents
    retriever = DocumentRetriever(documents)
    docs = retriever.retrieve_documents(question, final_top_n=5)

    # Format messages
    user_prompt = format_user_prompt(question, choices, docs)
    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    try:
        # Query model
        resp = client.chat.completions.create(
            messages = messages,
            model = "llama3p1-8b-instruct",
            temperature=0
        )

        # Extract answer
        model_answer = extract_answer(resp.choices[0].message.content)
        return model_answer
    except Exception as e:
        print(f"Query failed for question {question}: {e}")
        return None

def answer_query(question: str, choices: List[str], documents: List[str]) -> str:
    """
    Wrapper around solve. Converts given documents into dictionaries for compatibility with
    rest of script.

    Args:
        question: The obscure question
        choices: The multiple choice options
        documents: List of articles' text
    
    Returns:
        The model's answer to the given question.

    """
    documents = [{'title': f"Article {i}", 'text': doc} for i, doc in enumerate(documents)]
    return solve(question, choices, documents)


if __name__ == "__main__":
    # Load datasets
    wiki = load_dataset("nuprl/engineering-llm-systems", 
                        name="wikipedia-northeastern-university", 
                        split="test")
    
    questions = load_dataset("nuprl/engineering-llm-systems", 
                            name="obscure_questions", 
                            split="tiny")
    
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

        model_answer = solve(client, prompt, choices, wiki)

        if not model_answer:
            print(f"Failed to extract answer for question: {i}")
        
        # Check answer
        if model_answer == answer:
            num_correct += 1
            solved_questions.append(i)

    print(f"Solved {num_correct} questions")
    print(f"Solved questions: {solved_questions}")