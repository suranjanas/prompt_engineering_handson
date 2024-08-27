from dotenv import load_dotenv
from genai import Credentials, Client
# from genai.text.generation import TextGenerationParameters
from genai.schema import TextGenerationParameters
from ibm_watsonx_ai.foundation_models import ModelInference
import json
import requests
from tqdm import tqdm
import os


class BAM_LLM:
    def __init__(self, source="watsonx"):
        load_dotenv()

        self.source = source

        # WatsonX
        if self.source == "watsonx":
            load_dotenv()
            self.credentials = {
                "url": "https://us-south.ml.cloud.ibm.com",
                "apikey": os.getenv("WATSONX_KEY", None)
            }
            self.project_id = os.getenv("WATSONX_PROJECTID")


        credentials = Credentials.from_env()
        self.client = Client(credentials=credentials)

    def get_parameter_obj(self, params):
        if self.source == "bam": # For bam convert to TextGenerationParameters object
            parameters = TextGenerationParameters(**params)
        else: # For watsonx use dictionary as it is.
            parameters = params
        return parameters
    
    def generate_text(self, prompts, model_id, parameters):
        # WATSONX CALL
        if self.source == "watsonx":
            model = ModelInference(
                model_id=model_id, 
                params=parameters, 
                credentials=self.credentials,
                project_id=self.project_id
            )

            responses = model.generate(prompts)
            # with open("temp_output.json") as f:
            #     responses = json.load(f)
            generated_texts = []
            for response in responses:
                generated_texts.append(response["results"][0]["generated_text"])
            
            return generated_texts

        # BAM CALL
        if model_id != "ibm/granite-20b-code-instruct-v2":
            responses = list(
                self.client.text.generation.create(
                    model_id=model_id,
                    inputs=prompts,
                    parameters=parameters
                )
            )

            generated_texts = []
            for response in responses:
                generated_texts.append(response.results[0].generated_text)

            return generated_texts
        else:
            generated_texts = []
            for prompt in tqdm(prompts):
                payload={
                "input_cobol": prompt,
                "model_name": "instruct_v2_float16",
                "repetition_penalty": 1,
                "max_new_tokens": 1024,
                "max_sequence_length": 8192,
                "dynamic_padding": "yes",
                "truncate_input": "no"
                }
                data_json = json.dumps(payload)
                headers = {"Content-type": "application/json"}
                response =requests.post("http://cccxc567.pok.ibm.com:8000/llm_text_generation", data=data_json, headers=headers)
                generated_texts.append(response.json()['response'])
            
            return generated_texts
