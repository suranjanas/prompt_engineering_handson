import json, os
from IPython.display import Image, display
from collections import defaultdict
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

from langchain_ollama import ChatOllama


llm = ChatOllama(
    model = "llama3.1",
    base_url = "http://localhost:11434",
    temperature=0.9, num_predict=64)


   

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



#You can precise the format here which could be helpfull for multimodal graphs
class AgentState(TypedDict):
    input: str
    output: str
    decision: str

#Here is a simple 3 steps graph that is going to be working in the bellow "decision" condition
def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("get_input", get_user_input)
    #workflow.add_node("process_question", process_question)
    workflow.add_node("analyze", analyze_question)
    workflow.add_node("code_agent", answer_code_question)
    workflow.add_node("functionality_agent", code_functionality)
    workflow.add_node("task_fuction_matcher_agent", task_functionality_matcher)
    workflow.add_node("error_explanation_agent", generate_error_explanation)
    workflow.add_node("refinement_agent", refine_code)
    workflow.add_node("generic_agent", answer_generic_question)


    workflow.add_conditional_edges(
        "get_input",
        lambda x: "continue" if x["continue_conversation"] else "end",
        {
            "continue": "analyze",
            "end": END
        }
    )
    
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
            "yes": "get_input"
        }
    )
    workflow.set_entry_point("get_input")
    #workflow.set_entry_point("analyze")
    #workflow.add_edge("get_input", "analyze")
    #workflow.add_edge("process_question", "get_input")
    workflow.add_edge("code_agent", "functionality_agent")
    workflow.add_edge("functionality_agent", "task_fuction_matcher_agent")
    workflow.add_edge("error_explanation_agent", "refinement_agent")
    workflow.add_edge("refinement_agent", "get_input")
    workflow.add_edge("generic_agent", "get_input")
    #workflow.add_edge("get_input", END)
    #workflow.add_edge("refinement_agent", END)
    #workflow.add_edge("generic_agent", END)
    #workflow.add_edge("task_fuction_matcher_agent", END)

    return workflow.compile()



#Embedding API keys directly in your code is definilty not secure or recommended for production environments.
#Always use proper key management practices.


class UserInput(TypedDict):
    input: str
    continue_conversation: bool

def get_user_input(state: UserInput) -> UserInput:
    user_input = input("\nEnter your question (press 'q' to quit) : ")
    continue_conversation = True if user_input.lower() not in ['q', 'quit', 'bye', 'exit'] else False
    return {
        "input": user_input,
        "continue_conversation": continue_conversation
    }

def process_question():#state: UserInput):
    graph = create_graph()
    display(Image(graph.get_graph().draw_png()))
    #result = graph.invoke("input": state["input"]})
    result = graph.invoke({"input": "", "continue_conversation": True})
    return result

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
    # conversation_graph = create_conversation_graph()
    # display(Image(conversation_graph.get_graph().draw_png()))
    # conversation_graph.invoke({"input": "", "continue_conversation": True})
    result = process_question()
    try:
        if result["input"].lower() in ['q','quit','bye','exit']:
            print("Bye, see you again!!")
    except:
        print("\n--- Final answer ---")
        print(result["output"])


if __name__ == "__main__":
    main()