from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langserve import add_routes
from poke_api import PokeAPI
from pokemon_chain import get_battle_commentary

app = FastAPI(
    title="Pokemon Battle",
    description="A simple API for a pokemon battle",
    version="0.1",
)

class BattleInput(BaseModel):
    pokemon1_name: str
    # pokemon1_moves: dict[str, str] = {
    #     "move_one": str,
    #     "move_two": str,
    #     "move_three": str,
    # }
    pokemon2_name: str
    # pokemon2_moves: dict[str, str] = {
    #     "move_one": str,
    #     "move_two": str,
    #     "move_three": str,
    # }

@app.post("/battle")
async def battle(input: BattleInput):
    pokeAPI = PokeAPI()

    pokemon1_data = {
        "name": input.pokemon1_name,
        "moves": pokeAPI.get_moves(input.pokemon1_name),
        "stats": pokeAPI.get_stats(input.pokemon1_name),
        "abilities": pokeAPI.get_abilities(input.pokemon1_name),
    }

    pokemon2_data = {
        "name": input.pokemon2_name,
        "moves": pokeAPI.get_moves(input.pokemon2_name),
        "stats": pokeAPI.get_stats(input.pokemon2_name),
        "abilities": pokeAPI.get_abilities(input.pokemon2_name),
    }

    # if not pokemon1_data or not pokemon2_data:
    #     raise HTTPException(status_code=404, detail="Pokemon not found")

    # Format the data for langchain
    pokemon1_data_str = f"{pokemon1_data['name']} has the following moves: {pokemon1_data['moves']}. Their stats are: {pokemon1_data['stats']}. Their abilities are: {pokemon1_data['abilities']}"

    pokemon2_data_str = f"{pokemon2_data['name']} has the following moves: {pokemon2_data['moves']}. Their stats are: {pokemon2_data['stats']}. Their abilities are: {pokemon2_data['abilities']}"
    
    # get response from the langchain
    response = get_battle_commentary(pokemon1_data_str, pokemon2_data_str)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
