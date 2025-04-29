import sys
from InputPayload import InputPayload
from AgentController import AgentController
from DocumentSummerizationSkill import DocumentSummarySkill

document_payload = InputPayload(
    data = """
    Steel sales have been growing consistently over the past year. 
    Recent reports indicate a significant increase in revenue due to rising demand. 
    Market analysts predict further expansion in the steel industry. 
    However, global economic uncertainties could impact future growth. 
    Government policies are also expected to influence the steel sector. 
    Investment in infrastructure projects is one key driver for the increase in sales.
    """,
    data_type="text"
)



if __name__ == "__main__":
    args = sys.argv[1:]
    argument = ""

    # handling input
    if len(args) > 0:
        for arg in args:
            print(f"Argument: {arg}")
            argument = arg
            break
    else:
        print("No arguments provided.")
        exit()

    agent = AgentController()

    # # Register Skills
    agent.register_skill("summerize_content", DocumentSummarySkill())

    if(argument == "summerize_content"):
        print("In right path - ", argument)
        # Example Document
        summary = agent.handle_input("summerize_content", document_payload)
        print("\nSummary of the Document:")
        print("\n")
        print(summary)  
       


    

# # Generate and print summary
# print("Sentences -")
# print(document_text)

# summary = summarize_text(document_text, top_n=3)
# print("\n Summary of the Document:")
# print(summary)    
