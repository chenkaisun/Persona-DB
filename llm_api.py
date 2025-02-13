import openai
import time
from openai import OpenAI
from utils.utils import batchify
# make the above into a function
def chatbot(input, model="gpt-3.5-turbo-0613", response_format="text", batch_size=1500, seed=42):


    openai.organization = ""
    openai.api_key = ""
    # try until get a response
    client = OpenAI(
        api_key="",
        organization=""
    )
    output=None
    if "gpt" in model:

        while True:
            try:
                result = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input}
                    ],
                    response_format={"type": response_format},
                    seed=seed,
                    temperature=1,
                    # max_tokens=4096
                )
                # print("waiting half seconds")
                # time.sleep(1)
                output = result.choices[0].message.content
                break
            except Exception as e:  # wait 1 second and try again
                print(e)
                if e.type == "invalid_request_error":
                    break
                time.sleep(2)
                print("waiting 2 seconds")
        # output = result['choices'][0]['message']['content']

    else:

        input_batches = batchify(input, batch_size)
        res=[]
        while True:
            try:
                for ib in input_batches:
                    result = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=ib,
                    )
                    res.append(result)
                break
            except Exception as e:  # wait 1 second and try again
                print(e)
                print("waiting 1 second")
                time.sleep(1)
        output = [dat.embedding for obj in res for dat in obj.data]

    return output
