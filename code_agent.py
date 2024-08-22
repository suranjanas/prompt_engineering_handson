import json
from IPython.display import Image, display
from collections import defaultdict

from langchain_ollama import ChatOllama


llm = ChatOllama(
    model = "llama3.1",
    base_url = "http://localhost:11434",
    temperature=0.9, num_predict=64)

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

#global variable for code refinement
interm_values_list = []
class interm_values():
    def __init__(self):
        interm_values = {}
        interm_values["task_desc"] = ""
        interm_values["gen_code"] = ""
        interm_values["functionality"] = ""
        interm_values["matcher"] = ""
        interm_values["error_analysis"] = ""
        interm_values["refined_code"] = ""
    def set_task_desc(self,task_desc):
        interm_values["task_desc"] = task_desc
    def set_gen_code(self,gen_code):
        interm_values["gen_code"] = gen_code
    def set_functionlity(self,functionality):
        interm_values["functionality"] = functionality
    def set_matcher(self,matcher):
        interm_values["matcher"] = matcher
    def set_error_analysis(self,error_analysis):
        interm_values["error_analysis"] = error_analysis
    def set_refined_code(self,refined_code):
        interm_values["refined_code"] = refined_code
    def get_task_desc(self):
        return interm_values["task_desc"]
    def get_gen_code(self):
        return interm_values["gen_code"]
    def get_functionlity(self):
        return interm_values["functionality"]
    def get_matcher(self):
        return interm_values["matcher"]
    def get_error_analysis(self):
        return interm_values["error_analysis"]
    def get_refined_code(self):
        return interm_values["refined_code"]
    

# Creating the first analysis agent to check the prompt structure
# This print part helps you to trace the graph decisions
def analyze_question(state):
    #llm = ChatOpenAI()
    print("inside analyze question")
    llm = ChatOllama(
        model = "llama3.1",
        base_url = "http://localhost:11434",
        temperature=0.9, num_predict=64)
    prompt = PromptTemplate.from_template("""
    You are an agent that needs to define if a question is a technical code one or a general one.

    Question : {input}

    Analyse the question. Only answer with "code" if the question is about technical development. If not just answer "general".

    Your answer (code/general) :
    """)
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    decision = response.content.strip().lower()
    return {"decision": decision, "input": state["input"]}

# Creating the code agent that could be way more technical
def answer_code_question(state):
    print("inside answer code question")
    #llm = ChatOpenAI()
    llm = ChatOllama(
        model = "llama3.1",
        base_url = "http://localhost:11434",
        temperature=0.9, num_predict=64)
    prompt = PromptTemplate.from_template(
        "You are a software engineer. Answer this question with step by steps details : {input}"
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    output_dict = defaultdict(dict)
    output_dict = {"task_desc":state["input"],"code": response}
    return {"output": output_dict}

# Creating the SE agent that could generate code functionality
def code_functionality(state):
    #llm = ChatOpenAI()
    #input_dict = state["input"]
    print("code functionality ")
    llm = ChatOllama(
        model = "llama3.1",
        base_url = "http://localhost:11434",
        temperature=0.9, num_predict=64)
    prompt = PromptTemplate.from_template(
        "You are a software engineer. Observe the code given below and write the functionality of the code in 1 or 2 English sentences : {input}"
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["output"]["code"]})
    output_dict = defaultdict(dict)
    output_dict["task_desc"] = state["output"]["task_desc"]; output_dict["code"] = state["output"]["code"]; output_dict["functionality"] = response
    #output = {"task_desc":state["task_desc"],"code": state["input"],"output": response}
    return {"output": output_dict}


# Creating the SE agent that could match code functionality with task
def task_functionality_matcher(state):
    #llm = ChatOpenAI()
    print("matcher input:")
    llm = ChatOllama(
        model = "llama3.1",
        base_url = "http://localhost:11434",
        temperature=0.9, num_predict=64)
    prompt = PromptTemplate.from_template(
        "You are a software engineer. Observe the code functionality given below and the problem statement and decide if the functionality is matching the task description\
        or not. Answer in yes or no. do not generate anything else.\n\nCODE FUNCTIONALITY:\n{input1}\n\nTASK DESCRIPTION:\n{input2}\n\nANSWER:\n"
    )
    chain = prompt | llm
    matcher = chain.invoke({"input1": state["output"]["functionality"], "input2":state["output"]["task_desc"]})
    output_dict = defaultdict(dict)
    output_dict = {"matcher": matcher, "task_desc":state["output"]["task_desc"],"code": state["output"]["code"]}
    return {"output": output_dict}

def generate_error_explanation(state):
    print("error explanation")
    llm = ChatOllama(
        model = "llama3.1",
        base_url = "http://localhost:11434",
        temperature=0.9, num_predict=64)
    prompt = PromptTemplate.from_template(
        "You are a software engineer. Observe the generated code below for the given task, and explain the error in the code in 1 or 2 English sentences.\
        \n\nTASK DESCRIPTION:\n{input1}\n\nGENERATED CODE:\n{input2}\n\nERROR EXPLANATION:\n"
    )
    chain = prompt | llm
    response = chain.invoke({"input1": state["output"]["task_desc"], "input2": state["output"]["code"]})
    output = {}
    output = {"task_desc":state["output"]["task_desc"],"code": state["output"]["code"],"error_explanation": response}
    return {"output": output}

def refine_code(state):
    print("refine code")
    llm = ChatOllama(
        model = "llama3.1",
        base_url = "http://localhost:11434",
        temperature=0.9, num_predict=64)
    prompt = PromptTemplate.from_template(
        "You are a software engineer. Refine the buggy code given below which is to solve the given taskdescription, using the error explanation.\
        \n\nTASK DESCRIPTION:\n{input1}\n\nBUGGY CODE:\n{input2}\n\nERROR EXPLANATION:\n{input3}\n\nREFINED CODE:\n"
    )
    chain = prompt | llm
    response = chain.invoke({"input1":state["output"]["code"],"input2":state["output"]["code"],"input3": state["output"]["error_explanation"]})
    return {"output": response}

# Creating the generic agent
def answer_generic_question(state):
    print("answer generic question")
    #llm = ChatOpenAI()
    llm = ChatOllama(
        model = "llama3.1",
        base_url = "http://localhost:11434",
        temperature=0.9, num_predict=64)
    prompt = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input}"
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    return {"output": response}


from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
#from agents import analyze_question, answer_code_question, answer_generic_question

#You can precise the format here which could be helpfull for multimodal graphs
class AgentState(TypedDict):
    input: str
    output: str
    decision: str

#Here is a simple 3 steps graph that is going to be working in the bellow "decision" condition
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_question)
    workflow.add_node("code_agent", answer_code_question)
    workflow.add_node("functionality_agent", code_functionality)
    workflow.add_node("task_fuction_matcher_agent", task_functionality_matcher)
    workflow.add_node("error_explanation_agent", generate_error_explanation)
    workflow.add_node("refinement_agent", refine_code)
    workflow.add_node("generic_agent", answer_generic_question)

    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "code": "code_agent",
            "general": "generic_agent"
        }
    )
    workflow.add_conditional_edges(
        "task_fuction_matcher_agent",
        lambda x: "no" if x["output"]["matcher"].content.lower().strip()=="no" else "yes",
        {
            "no": "error_explanation_agent",
            "yes": END
        }
    )

    workflow.set_entry_point("analyze")
    workflow.add_edge("code_agent", "functionality_agent")
    workflow.add_edge(["functionality_agent"], "task_fuction_matcher_agent")
    workflow.add_edge("error_explanation_agent", "refinement_agent")
    workflow.add_edge("refinement_agent", END)
    workflow.add_edge("generic_agent", END)
    workflow.add_edge("task_fuction_matcher_agent", END)

    return workflow.compile()


import os

#Embedding API keys directly in your code is definilty not secure or recommended for production environments.
#Always use proper key management practices.
os.environ["OPENAI_API_KEY"] = "your-openai-key"

#from graph import create_graph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

class UserInput(TypedDict):
    input: str
    continue_conversation: bool

def get_user_input(state: UserInput) -> UserInput:
    user_input = input("\nEnter your question (ou 'q' to quit) : ")
    return {
        "input": user_input,
        "continue_conversation": user_input.lower() != 'q'
    }

def process_question(state: UserInput):
    graph = create_graph()
    result = graph.invoke({"input": state["input"]})
    print("\n--- Final answer ---")
    print(result["output"])
    return state

def create_conversation_graph():
    workflow = StateGraph(UserInput)

    workflow.add_node("get_input", get_user_input)
    workflow.add_node("process_question", process_question)

    workflow.set_entry_point("get_input")

    workflow.add_conditional_edges(
        "get_input",
        lambda x: "continue" if x["continue_conversation"] else "end",
        {
            "continue": "process_question",
            "end": END
        }
    )

    workflow.add_edge("process_question", "get_input")

    return workflow.compile()

def main():
    conversation_graph = create_conversation_graph()
    display(Image(conversation_graph.get_graph().draw_png()))
    conversation_graph.invoke({"input": "", "continue_conversation": True})

if __name__ == "__main__":
    main()