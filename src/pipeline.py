from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str)-> dict:
    state = {}

    # Step 1 - Search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content

    print("\n search result",state['search_results'])


    # Step 2 - reader agent
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
         "messages": [("user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content
    print("\nScraped content",state['scraped_content'])


    # Step 3 - Writer Chain
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    # There are two agents, first one scraps the web regarding that information and then the second agent basically reads the data properly and answer fully.
    # Now ↓ here we are just combining both the results that we are getting from the agent 1 and agent 2.
    research_combined = (
        f"SEARCH RESULTS: \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT: \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined 
    })

    print(['\n Final Report\n', state['report']])

    # critic report
    print("\n"+" ="*50)
    print("Step 4 - Critic is reviewing the report")
    print("="*50)

    state['feedback'] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state

if __name__ == "__main__":
    topic = input("\n Enter a research topic: ")
    run_research_pipeline(topic)



