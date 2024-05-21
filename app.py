import os

#from langchain.docstore.document import Document
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
#from langchain_community.vectorstores import Chroma
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def invoke_chain(code):

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                            google_api_key=GOOGLE_API_KEY,
                             temperature=0.3)

    # section for generating automation test
    test_prompt = PromptTemplate.from_template("""You are an expert automation tester.
                                            Your job is to write the junit test cases for the below code.\n
                                            Code: {code} \nAnswer:
                                        """)
    
    test_chain = LLMChain(llm=model, prompt=test_prompt,output_key="test")
    test_response = test_chain.invoke({"code": code})
    # section for generating test coverage
    coverage_prompt = PromptTemplate.from_template("""
                                            You are an expert in testing frameworks.
                                            Execute the test cases with respect to the below code using a standard test coverage framework suitable for this programming language.
                                            Based on this create a short and concise test coverage report using that test coverage framework.\n
                                            Code: {code}\n
                                            Test cases: {test} \nAnswer:
                                        """)
    
    coverage_chain = LLMChain(llm=model, prompt=coverage_prompt, output_key="coverage_report")
    coverage_response = coverage_chain.invoke({"code": test_response["code"],"test": test_response["test"]})
    # section for coverage analysis
    analysis_prompt = PromptTemplate.from_template("""
                                            Your job is to analyze the coverage report provided and provide an ouput with the below conditions\n
                                            1. If the coverage is 100 percent return only '100',nothing else
                                            2. If the coverage is not 100 percent return the uncovered functions or statements.\n

                                            Coverage report: {report}\n
                                            Answer:
                                        """)
    
    analysis_chain = LLMChain(llm=model, prompt=analysis_prompt, output_key="coverage_analysis")
    analysis_response = analysis_chain.invoke({"report":coverage_response["coverage_report"]})
    coverage_response["coverage_analysis"] = analysis_response["coverage_analysis"]
    coverage_response["pre_generated_tests"] = coverage_response["test"]
    # if coverage is not 100%
    if coverage_response["coverage_analysis"]!="100":
        regenerate_prompt = PromptTemplate.from_template("""
                                            The below test cases have failed to cover the below code.
                                            Your job is to analysis the coverage report and generate more accurate test cases along with original one to increase the coverage percentage to 100.\n
                                                         
                                            Code: {code}\n
                                            Original Test Cases: {test}\n
                                            Coverage report: {report}\n
                                            Answer:
                                        """)
    
        chain = LLMChain(llm=model, prompt=regenerate_prompt, output_key="test")
        response = chain.invoke({
            "code":coverage_response["code"],
            "test":coverage_response["test"],
            "report":coverage_response["coverage_report"]
            })
        coverage_response["test"]=response["test"]
        # now we again generate a coverage report
        response = coverage_chain.invoke({"code": coverage_response["code"],"test": coverage_response["test"]})
        coverage_response["coverage_report"]=response["coverage_report"]
        # now we again do the analysis
        response = analysis_chain.invoke({"report":coverage_response["coverage_report"]})
        coverage_response["coverage_analysis"] = response["coverage_analysis"]

    
    return coverage_response




st.set_page_config("Create automation tasks using GEN AI")
st.header("Create automation tasks using GEN AI!!!💁")
code = st.text_area('Upload your code...')
submit = st.button('Submit')

if code and submit:
    answer=invoke_chain(code)
    st.write(answer)