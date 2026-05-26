"""
Test LLM connection and demonstrate LCEL chains.
Run: python -m backend.test_llm
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from backend.llm_config import get_llm


def test_1_basic_call():
    """Test: Send messages to the LLM and get a response."""
    print("Test 1: Basic LLM call")
    print("-" * 40)

    llm = get_llm()

    # Messages are a list of message objects
    # The LLM processes the entire conversation history
    messages = [
        SystemMessage(content="You are AMIA, a research assistant. Be concise."),
        HumanMessage(content="What is LangGraph in one sentence?"),
    ]

    # .invoke() sends the messages and returns an AIMessage object
    response = llm.invoke(messages)
    # response.content is the text string of the response

    print(f"Response: {response.content}")
    print(f"Type: {type(response)}")  # AIMessage
    print("✅ Basic call working\n")


def test_2_lcel_chain():
    """Test: LCEL chain — prompt | llm | parser."""
    print("Test 2: LCEL chain")
    print("-" * 40)

    llm = get_llm()

    # ChatPromptTemplate creates a reusable message template
    # {topic} is a placeholder that gets filled by .invoke()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise research assistant."),
        ("human", "Explain {topic} in exactly 2 sentences."),
    ])

    # StrOutputParser extracts just the .content string from the AIMessage
    parser = StrOutputParser()

    # Chain them with | (pipe)
    # The data flows: dict → prompt → llm → parser → str
    chain = prompt | llm | parser

    # .invoke() runs the entire chain
    result = chain.invoke({"topic": "embeddings in AI"})
    # result is now a plain string (not an AIMessage)

    print(f"Result: {result}")
    print(f"Type: {type(result)}")  # str
    print("✅ LCEL chain working\n")


def test_3_structured_output():
    """Test: Force the LLM to return structured JSON."""
    print("Test 3: Structured JSON output")
    print("-" * 40)

    llm = get_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the query. Return ONLY valid JSON, no other text:\n"
                   '{{"complexity": "low|medium|high", '
                   '"needs_web_search": true|false, '
                   '"needs_vector_db": true|false}}'),
        ("human", "Query: {query}"),
    ])

    # JsonOutputParser parses the LLM's text into a Python dict
    parser = JsonOutputParser()

    chain = prompt | llm | parser

    result = chain.invoke({"query": "What happened in AI research last week?"})
    print(f"Result: {result}")
    print(f"Type: {type(result)}")  # dict
    print(f"Complexity: {result.get('complexity')}")
    print("✅ Structured output working\n")


def test_4_streaming():
    """Test: Stream tokens one by one (for real-time UI)."""
    print("Test 4: Streaming")
    print("-" * 40)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Count from 1 to 5, one number per line."),
    ])
    chain = prompt | llm | StrOutputParser()

    # .stream() returns a generator — each iteration yields one token
    print("Streaming: ", end="", flush=True)
    for chunk in chain.stream({}):
        print(chunk, end="", flush=True)
    print("\n✅ Streaming working\n")


if __name__ == "__main__":
    test_1_basic_call()
    test_2_lcel_chain()
    test_3_structured_output()
    test_4_streaming()
    print("🎉 All LLM tests passed!")