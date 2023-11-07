import re
import ast
import spacy
import openai
import argparse
import tiktoken
import pandas as pd
from scipy import spatial  # for calculating vector similarities for search

openai.api_key = "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"

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
# Load the Italian language model
nlp = spacy.load('it_core_news_sm')
df3 = pd.read_csv("../data/embedded_stip.csv")
# convert embeddings from CSV str type back to list type
df3['embedding_text_answer'] = df3['embedding_text_answer'].apply(ast.literal_eval)

INTRODUCTION = """
        Sei un chatbot gentile che risponde ai clienti di Lavazza per assisterli sui suoi prodotti.
        Per rispondere, utilizza le seguenti coppie di domande e risposte fatte dai clienti algi assistenti che lavorano per l'azienda Lavazza. 
        Se hai a disposizione molte informazioni, preferisci rispondere basandoti sulla data più recente.
        Sii il più specifico possibile, fornisci links e/o numero di telefono se necessario.
        Se non trovi una risposta specifica vai sul sito ufficiale "https://www.lavazza.it/it/" e cerca il prodotto che viene richiesto, fornendo il link completo al cliente.
        Se la domanda/affermazione fa emergere delle complicanze rispondi scusandoti ed invitando a contattare il servizio clienti al 
        Numero Verde 800124535 o all'indirizzo e-mail info@lavazza.it"
        Se non trovi risposta rispondi 'Non trovo una risposta, puoi essere più specifico?' 
        """


def preprocess_text(text: str, answer: bool = False, train_data: bool = False):
    if answer:
        text = re.sub(r'Ciao ({customer_name}|\w+)(?: |,)', 'Ciao', text)
        return text

    # Remove emoticons
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    text = re.sub(r'http\S+', 'url', text)  # Replace or remove URLs
    # Replace or remove email addresses
    text = re.sub(r'\S+@\S+', 'email', text)

    # Remove unnecessary characters and whitespace
    # replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # clean up multiple spaces
    text = re.sub(r'x|X', 'per', text)

    if train_data:
        if text.startswith("Lavazza"):
            text = text[8:]

        names = []
        # Process the text with the NER model
        doc = nlp(text)
        for entity in doc.ents:
            if entity.label_ == "PER" and text.startswith(entity.text) and len(entity.text.split()) > 1 and "Lavazza" not in entity.text:
                text = re.sub(entity.text, '', text)
                names.append(entity.text)
        if names:
            print(names)

    return text.lower().strip()


# search function
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100) -> tuple[list[str], list[float]]:
    """
    Returns a list of strings and relatednesses, 
    sorted from most related to least.
    """

    query = preprocess_text(query)
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["Data"], row["Testo"], row["Risposta"], relatedness_fn(
            query_embedding, row["embedding_text_answer"]))
        for _, row in df.iterrows()
    ]

    strings_and_relatednesses.sort(key=lambda x: x[3], reverse=True)
    data, testo, risposta, relatednesses = zip(*strings_and_relatednesses)

    return data[:top_n], testo[:top_n], risposta[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:

    """
    Return a message for GPT, 
    with relevant source texts pulled from a dataframe.
    """

    dates, strings, answers, relatednesses = strings_ranked_by_relatedness(
        query, df)
    # print(strings[0])
    # print(answers[0])
    

    question = f"\n\nDomanda: {query}"
    message = INTRODUCTION

    # for string in strings:
    distinct_answers = set()
    for data, string, answer in zip(dates, strings, answers):
        if (
            num_tokens(message + f"Data: {data}" + f"n\Domanda: {string}" +
                       f"\nRisposta: {answer}\n" + question, model=model)
            > token_budget
        ) or answer.lower() in distinct_answers:
            break
        else:
            message += "\n" + \
                f"Data: {data}" + f"\nDomanda: {string}" + \
                f"\nRisposta: {answer}\n"
            distinct_answers.update(answer.lower())
    return message + "\n" + question


def ask_gpt(
    query: str,
    fine_tuning: bool = False,
    df: pd.DataFrame = df3,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""

    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)

    model_= "ft:gpt-3.5-turbo-0613:stip:lavazza-ft:8G9cCh9V"
    messages = [
        {"role": "system", "content": "Sei un chatbot che risponde ai clienti di Lavazza per assisterli sui suoi prodotti."},
        {"role": "user", "content": message if not fine_tuning else INTRODUCTION + "\n" + query},
    ]

    response = openai.ChatCompletion.create(
        model=model if not fine_tuning else model_,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


def main(Q: str,
         gpt4: bool,
         fine_tuning: bool,
         print_: bool):
    
    #print(gpt4, fine_tuning)
    assert Q is not None, "Insert a Query please"
    #while Q.lower() != "stop":
    if fine_tuning:
        print(ask_gpt(query=Q,fine_tuning=fine_tuning, print_message=print_))
    else:
        if gpt4: 
            print(ask_gpt(query=Q, print_message=print_, model="gpt-4"))
        else: 
            print(ask_gpt(query=Q, print_message=print_))
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Given a query, openAI API provides an answer")
    parser.add_argument(
        "-q", "--query", help="Print ranked messages")
    parser.add_argument(
        "-gpt4", "--gpt4", help="chose between gpt3.5 and gpt4", action="store_true")
    parser.add_argument(
        "-ft", "--fine_tuning", help="Use Fine Tune Model instead of QA one", action="store_true")
    parser.add_argument(
        "-p", "--prints", help="Print ranked messages", action="store_true")

    args = parser.parse_args()
    main(args.query,
         args.gpt4,
         args.fine_tuning,
         args.prints)
