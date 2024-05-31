# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from poke_api import PokeAPI # type: ignore
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# lang smith api key
# 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

# User input GET POKEMON DATA ----------------
pokemon_one = input("Pick the first pokemon: ")
pokemon_two = input("Pick the second pokemon: ")
pokeAPI = PokeAPI()

pokemon_one_obj = {
    "name": pokemon_one,
    "moves": pokeAPI.get_moves(pokemon_one),
    "stats": pokeAPI.get_stats(pokemon_one),
    "abilities": pokeAPI.get_abilities(pokemon_one),
    # "weaknesses": pokeAPI.get_weaknesses(pokemon_one),
    # "resistances": pokeAPI.get_resistances(pokemon_one)
}

print(f"{pokemon_one_obj['name']} has the following moves: {pokemon_one_obj['moves']}")
pokemon_two_obj = {
    "name": pokemon_two,
    "moves": pokeAPI.get_moves(pokemon_two),
    "stats": pokeAPI.get_stats(pokemon_two),
    "abilities": pokeAPI.get_abilities(pokemon_two),
    # "weaknesses": pokeAPI.get_weaknesses(pokemon_two),
    # "resistances": pokeAPI.get_resistances(pokemon_two)
}

print(f"{pokemon_two_obj['name']} has the following moves: {pokemon_two_obj['moves']}")

# CREATE CLIENT ------------------------
# Point to the local server
system_template = "You are the number one pokemon battle referee and have the knowledge of all pokemon. You offer detailed play by play commentary of pokemon battles in a clear, impartial and engaging way. You will keep track of each pokemons health and attack and once a pokemon reaches 0 health you will end the battle. Please keep your responses focused on the simulated battle at hand. After each move you will ask the user what you want to do next providing four options for their next move alternating between each pokemon giving them an option to do something."

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("assistant", f"The two pokemons selected to battle are {pokemon_one_obj['name']} and {pokemon_two_obj['name']}"),
    ("system", f"Because you are the best pokemon referee, you know all about {pokemon_one_obj['name']} and its data which is {pokemon_one_obj}. You also know all about {pokemon_two_obj['name']} and its data which is {pokemon_two_obj}. Let the battle begin! {pokemon_one_obj['name']} what is your first move?"),
])

# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = ChatOpenAI()

parser = StrOutputParser()
chain = prompt_template | model | parser

app = FastAPI(
    title="Pokemon Battle Referee",
    version="1.0",
    description="A chat bot that can simulate pokemon battles",
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

# history = [
#     {"role": "system", "content": "You are the number one pokemon battle referee and have the knowledge of all pokemon. You offer detailed play by play commentary of pokemon battles in a clear, impartial and engaging way. You will keep track of each pokemons health and attack and once a pokemon reaches 0 health you will end the battle. Please keep your responses focused on the simulated battle at hand. After each move you will ask the user what you want to do next providing four options for their next move alternating between each pokemon giving them an option to do something."},
#     {"role": "assistant", "content": f"The two pokemon selected to battle are {pokemon_one_obj["name"]} and {pokemon_two_obj["name"]}"},
#     {"role": "system", "content": f"Because you are the best pokemon referee you know all about {pokemon_one_obj["name"]} and it's data which is {pokemon_one_obj}. You also know all about {pokemon_two_obj["name"]} and it's data which is {pokemon_two_obj}. Let the battle begin! {pokemon_one_obj["name"]} what is your first move?"},
# ]

# while True:
#     completion = client.chat.completions.create(
#         model="FaradayDotDev/llama-3-8b-Instruct-GGUF",
#         messages=history,
#         temperature=0.7,
#         stream=True,
#     )

#     new_message = {"role": "assistant", "content": ""}
    
#     for chunk in completion:
#         if chunk.choices[0].delta.content:
#             print(chunk.choices[0].delta.content, end="", flush=True)
#             new_message["content"] += chunk.choices[0].delta.content

#     history.append(new_message)
    
#     # Uncomment to see chat history
#     # import json
#     # gray_color = "\033[90m"
#     # reset_color = "\033[0m"
#     # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
#     # print(json.dumps(history, indent=2))
#     # print(f"\n{'-'*55}\n{reset_color}")

#     print()
#     history.append({"role": "user", "content": input("> ")})
