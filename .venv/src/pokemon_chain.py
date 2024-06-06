from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# define prompt template
prompt_template = PromptTemplate(
    input_variables=["pokemon1_data", "move1_data", "pokemon2_data", "move2_data"],
    template="""
        Referee: A new Pokémon battle has started!
        Pokémon 1: {pokemon1_data}
        Pokémon 2: {pokemon2_data}
        Referee: Please comment on the battle status.
        """
    )

llm = OpenAI()

def get_battle_commentary(pokemon1_data, pokemon2_data):
    prompt = prompt_template.format(pokemon1_data=pokemon1_data, pokemon2_data=pokemon2_data)

    response = llm(prompt)
    return response
